//! Anthropic models served by Amazon Bedrock.
//!
//! Bedrock speaks the Messages API with a few differences: the model is in
//! the URL, `anthropic_version` replaces the version header, streaming uses
//! the AWS event stream encoding with each Anthropic event base64-encoded
//! inside a `chunk`, and errors use AWS envelopes.

use serde::Deserialize;

use super::stream::AnthropicStreamDecoder;
use crate::error::{Error, ErrorKind, Result};
use crate::http::{enrich_error_from_headers, error_kind_for_status};
use crate::protocols::{ApiProfile, StreamDecoder};
use crate::stream::{StreamEvent, StreamNormalizer};
use crate::transport::{HeaderMap, HeaderName};
use crate::util::base64_decode;

/// The body field Bedrock requires in place of the `anthropic-version` header.
pub(super) const ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

/// The header carrying the exception class on error responses, as
/// `ValidationException` or `ValidationException:http://...`.
pub(crate) const ERROR_TYPE: HeaderName = HeaderName::from_static("x-amzn-errortype");

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
fn bedrock_error_kind(exception: Option<&str>, status: u16) -> ErrorKind {
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
    #[serde(default, alias = "Message")]
    message: Option<String>,
}

fn exception_message(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<ExceptionBody>(body)
        .ok()
        .and_then(|body| body.message)
}

/// The message for a stream exception. The payload is a JSON envelope for
/// `:message-type: exception` but bare `:error-message` text for
/// `:message-type: error`, so both shapes have to be handled.
fn stream_exception_message(kind: &str, payload: &str) -> String {
    exception_message(payload.as_bytes())
        .or_else(|| {
            Some(payload.trim())
                .filter(|text| !text.is_empty())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| format!("bedrock stream failed with {kind}"))
}

pub(super) fn decode_bedrock_error(status: u16, headers: &HeaderMap, body: &[u8]) -> Error {
    let exception = headers
        .get(ERROR_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.split(':').next().unwrap_or(value).to_owned());
    let kind = bedrock_error_kind(exception.as_deref(), status);
    let message = exception_message(body).unwrap_or_else(|| {
        crate::protocols::fallback_provider_error_message(
            ApiProfile::BedrockAnthropic,
            status,
            body,
        )
    });
    let mut error = Error::new(kind, message).with_origin(ApiProfile::BedrockAnthropic.as_str());
    if let Some(exception) = exception {
        error = error.with_code(exception);
    }
    enrich_error_from_headers(error, status, headers)
}

/// The payload of a Bedrock `chunk` event.
#[derive(Deserialize)]
struct PayloadPart {
    bytes: String,
}

/// Unwraps Bedrock's stream envelope around the Anthropic events.
pub(super) struct BedrockStreamDecoder {
    inner: AnthropicStreamDecoder,
}

impl BedrockStreamDecoder {
    pub(super) fn new() -> Self {
        Self {
            inner: AnthropicStreamDecoder::new(ApiProfile::BedrockAnthropic),
        }
    }
}

impl StreamDecoder for BedrockStreamDecoder {
    fn on_frame(
        &mut self,
        data: String,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) -> Result<()> {
        let part: PayloadPart = serde_json::from_str(&data)
            .map_err(|error| Error::malformed(format!("bedrock: invalid stream chunk: {error}")))?;
        let bytes = base64_decode(&part.bytes, false)
            .ok_or_else(|| Error::malformed("bedrock: stream chunk is not valid base64"))?;
        let event = String::from_utf8(bytes)
            .map_err(|_| Error::malformed("bedrock: stream chunk is not valid UTF-8"))?;
        self.inner.on_frame(event, normalizer, out)
    }

    fn on_exception(
        &mut self,
        kind: String,
        payload: String,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let error_kind = bedrock_error_kind(Some(&kind), 200);
        let message = stream_exception_message(&kind, &payload);
        let error = Error::new(error_kind, message)
            .with_origin(ApiProfile::BedrockAnthropic.as_str())
            .with_code(kind);
        normalizer.fail(out, error);
    }

    fn on_eof(&mut self, normalizer: &mut StreamNormalizer, out: &mut Vec<StreamEvent>) {
        self.inner.on_eof(normalizer, out);
    }
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
    fn stream_error_messages_survive_a_non_json_payload() {
        assert_eq!(
            stream_exception_message("modelStreamErrorException", "the model stopped"),
            "the model stopped"
        );
        assert_eq!(
            stream_exception_message("throttlingException", r#"{"message":"slow down"}"#),
            "slow down"
        );
        assert_eq!(
            stream_exception_message("InternalFailure", "  "),
            "bedrock stream failed with InternalFailure"
        );
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
    fn error_type_header_drops_its_uri_suffix() {
        let headers = HeaderMap::from_iter([(
            ERROR_TYPE,
            "ValidationException:http://internal.amazon.com/coral/com.amazon.bedrock/"
                .parse()
                .unwrap(),
        )]);

        let error = decode_bedrock_error(400, &headers, br#"{"message":"bad max_tokens"}"#);

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(error.code(), Some("ValidationException"));
        assert_eq!(error.message(), "bad max_tokens");
    }
}
