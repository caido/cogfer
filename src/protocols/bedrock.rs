//! The Amazon Bedrock error envelope, shared by every profile Bedrock serves.
//!
//! Bedrock reports endpoint-level failures the AWS way regardless of the wire
//! format inside: the exception class arrives in the `x-amzn-errortype` header
//! and the body is a bare `{"message": ...}` envelope. The event-stream
//! framing used by `InvokeModelWithResponseStream` lives with its only
//! consumer in [`super::anthropic::bedrock`]; the OpenAI-compatible endpoints
//! stream plain Server-Sent Events.

use serde::Deserialize;
use url::Url;

use crate::error::{Error, ErrorKind};
use crate::http::{enrich_error_from_headers, error_kind_for_status, join_url};
use crate::protocols::ApiProfile;
use crate::transport::{HeaderMap, HeaderName, HttpRequest};

/// The header carrying the exception class on error responses, as
/// `ValidationException` or `ValidationException:http://...`.
pub(crate) const ERROR_TYPE: HeaderName = HeaderName::from_static("x-amzn-errortype");

/// The path of the OpenAI-compatible API under the runtime endpoint.
pub(crate) const OPENAI_PATH: &str = "/openai/v1";

/// `ListAsyncInvokes`: the one free read on the runtime endpoint, so it works
/// through a custom or VPC endpoint where a control-plane call would not, and
/// answers in every region.
pub(crate) fn verify_request(base_url: &Url) -> HttpRequest {
    let mut root = base_url.clone();
    let path = root.path().trim_end_matches('/');
    if let Some(prefix) = path.strip_suffix(OPENAI_PATH).map(str::to_owned) {
        root.set_path(&prefix);
    }
    let mut url = join_url(&root, "async-invoke");
    // One summary is enough to have the call authorized.
    url.query_pairs_mut().append_pair("maxResults", "1");
    HttpRequest::get(url)
}

/// Whether AWS blamed the credentials or the signature rather than the
/// caller's permissions, meaning a fresh signature could succeed.
///
/// [`bedrock_error_kind`] classifies these as [`ErrorKind::Authentication`]
/// and [`crate::aws::SigV4Authenticator`] retries exactly them, so the two
/// cannot disagree.
pub(crate) fn is_signature_failure(exception: &str) -> bool {
    let exception = exception.to_ascii_lowercase();
    [
        "signature",
        "token",
        "unrecognizedclient",
        "skew",
        "requestexpired",
    ]
    .iter()
    .any(|cause| exception.contains(cause))
}

/// Classify an AWS exception by its class name and HTTP status.
pub(crate) fn bedrock_error_kind(exception: Option<&str>, status: u16) -> ErrorKind {
    let exception = exception.unwrap_or_default().to_ascii_lowercase();
    let mentions = |needle: &str| exception.contains(needle);
    if is_signature_failure(&exception) {
        ErrorKind::Authentication
    } else if mentions("accessdenied") {
        ErrorKind::Permission
    } else if mentions("throttling") || mentions("quota") {
        ErrorKind::RateLimited
    } else if mentions("modelnotready") || mentions("serviceunavailable") {
        ErrorKind::Overloaded
    } else if mentions("timeout") {
        ErrorKind::Timeout
    } else if mentions("validation") {
        ErrorKind::InvalidRequest
    } else if mentions("resourcenotfound") {
        ErrorKind::NotFound
    } else {
        match status {
            408 => ErrorKind::Timeout,
            424 => ErrorKind::Provider,
            _ => error_kind_for_status(status),
        }
    }
}

#[derive(Deserialize)]
struct ExceptionBody {
    #[serde(alias = "Message")]
    message: Option<String>,
}

pub(crate) fn exception_message(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<ExceptionBody>(body)
        .ok()
        .and_then(|body| body.message)
}

/// Whether an `AccessDeniedException` is about the credentials themselves.
/// A Bedrock API key that is malformed, wrong or missing is refused this way
/// rather than with a signature exception, while a policy denial names the
/// action that is not authorized.
fn rejects_credentials(message: Option<&str>) -> bool {
    let message = message.unwrap_or_default().to_ascii_lowercase();
    message.contains("api key") || message.contains("authorization header")
}

/// Decode an AWS error envelope, attributed to `profile`.
pub(crate) fn decode_bedrock_error(
    profile: ApiProfile,
    status: u16,
    headers: &HeaderMap,
    body: &[u8],
) -> Error {
    let exception = headers
        .get(ERROR_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.split(':').next().unwrap_or(value).to_owned());
    let message = exception_message(body);
    let kind = match bedrock_error_kind(exception.as_deref(), status) {
        ErrorKind::Permission if rejects_credentials(message.as_deref()) => {
            ErrorKind::Authentication
        }
        kind => kind,
    };
    let message = message.unwrap_or_else(|| {
        crate::protocols::fallback_provider_error_message(profile, status, body)
    });
    let mut error = Error::new(kind, message).with_origin(profile.as_str());
    if let Some(exception) = exception {
        error = error.with_code(exception);
    }
    enrich_error_from_headers(error, status, headers)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every exception the retry predicate calls a signature failure must
    /// also be classified as `Authentication`, and no other one may be.
    #[test]
    fn signature_failures_are_exactly_the_authentication_exceptions() {
        for exception in [
            "InvalidSignatureException",
            "ExpiredTokenException",
            "UnrecognizedClientException",
            "RequestTimeTooSkewed",
            "RequestExpired",
        ] {
            assert!(is_signature_failure(exception), "{exception}");
            assert_eq!(
                bedrock_error_kind(Some(exception), 403),
                ErrorKind::Authentication,
                "{exception}"
            );
        }
        for exception in ["AccessDeniedException", "ThrottlingException"] {
            assert!(!is_signature_failure(exception), "{exception}");
            assert_ne!(
                bedrock_error_kind(Some(exception), 403),
                ErrorKind::Authentication,
                "{exception}"
            );
        }
    }

    #[test]
    fn exceptions_map_to_error_kinds() {
        assert_eq!(
            bedrock_error_kind(Some("ThrottlingException"), 429),
            ErrorKind::RateLimited
        );
        assert_eq!(
            bedrock_error_kind(Some("ExpiredTokenException"), 403),
            ErrorKind::Authentication
        );
        assert_eq!(
            bedrock_error_kind(Some("AccessDeniedException"), 403),
            ErrorKind::Permission
        );
        assert_eq!(
            bedrock_error_kind(Some("RequestTimeTooSkewed"), 403),
            ErrorKind::Authentication
        );
        assert_eq!(
            bedrock_error_kind(Some("modelStreamErrorException"), 200),
            ErrorKind::Provider
        );
        assert_eq!(bedrock_error_kind(None, 503), ErrorKind::Overloaded);
    }

    #[test]
    fn api_key_rejections_are_authentication_failures() {
        let headers =
            HeaderMap::from_iter([(ERROR_TYPE, "AccessDeniedException".parse().unwrap())]);
        let decode = |message: &str| {
            decode_bedrock_error(
                ApiProfile::BedrockAnthropic,
                403,
                &headers,
                format!(r#"{{"Message":"{message}"}}"#).as_bytes(),
            )
        };

        for message in [
            "Authentication failed: Please make sure your API Key is valid.",
            "Invalid API Key format: Must start with pre-defined prefix",
            "Authorization header is missing",
        ] {
            assert_eq!(
                decode(message).kind(),
                ErrorKind::Authentication,
                "{message}"
            );
        }
        let denied = decode(
            "User: arn:aws:iam::123456789012:user/caido is not authorized to perform: \
             bedrock:ListAsyncInvokes on resource: arn:aws:bedrock:us-east-1:123456789012:*",
        );
        assert_eq!(denied.kind(), ErrorKind::Permission);
        assert_eq!(denied.code(), Some("AccessDeniedException"));
    }

    #[test]
    fn verify_lists_async_invokes_on_the_runtime_root() {
        for (base, expected) in [
            (
                "https://bedrock-runtime.eu-west-1.amazonaws.com",
                "https://bedrock-runtime.eu-west-1.amazonaws.com/async-invoke?maxResults=1",
            ),
            (
                "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1",
                "https://bedrock-runtime.eu-west-1.amazonaws.com/async-invoke?maxResults=1",
            ),
            (
                "https://vpce-1.bedrock-runtime.eu-west-1.vpce.amazonaws.com/openai/v1/",
                "https://vpce-1.bedrock-runtime.eu-west-1.vpce.amazonaws.com/async-invoke?maxResults=1",
            ),
            (
                "https://gateway.example.com/bedrock/openai/v1",
                "https://gateway.example.com/bedrock/async-invoke?maxResults=1",
            ),
        ] {
            let request = verify_request(&Url::parse(base).unwrap());
            assert_eq!(request.method, crate::transport::Method::GET);
            assert_eq!(request.url.as_str(), expected, "{base}");
        }
    }

    #[test]
    fn error_type_header_drops_its_uri_suffix() {
        let headers = HeaderMap::from_iter([(
            ERROR_TYPE,
            "ValidationException:http://internal.amazon.com/coral/com.amazon.bedrock/"
                .parse()
                .unwrap(),
        )]);

        let error = decode_bedrock_error(
            ApiProfile::BedrockAnthropic,
            400,
            &headers,
            br#"{"message":"bad max_tokens"}"#,
        );

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.code(), Some("ValidationException"));
        assert_eq!(error.message(), "bad max_tokens");
    }
}
