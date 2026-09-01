//! AWS Signature Version 4 over an [`HttpRequest`].
//!
//! Signing is [`aws_sigv4`]. This module adapts our request to its input and
//! applies the headers it returns. Its defaults are what Bedrock expects:
//! header-based signatures, double percent-encoded paths (so a model ID's `:`
//! survives), and the session token folded into the signature.

use std::time::SystemTime;

use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
use aws_sigv4::sign::v4;

use super::AwsCredentials;
use crate::error::{Error, Result};
use crate::http::header_value;
use crate::transport::{HeaderName, HttpRequest, header};

const AMZ_DATE: HeaderName = HeaderName::from_static("x-amz-date");
const SECURITY_TOKEN: HeaderName = HeaderName::from_static("x-amz-security-token");

/// Headers a previous signature leaves behind. Removing them first makes
/// re-signing produce the same result as signing once.
const GENERATED: [HeaderName; 3] = [header::AUTHORIZATION, AMZ_DATE, SECURITY_TOKEN];

/// Sign `request` in place, replacing any previous signature.
///
/// The body must be final: its hash is part of the signature. Signing sets
/// `host`, `x-amz-date`, `authorization`, and `x-amz-security-token` for
/// temporary credentials.
///
/// # Errors
///
/// Returns an error when the URL cannot be signed or a header the signer
/// produced is not a valid header value.
pub fn sign_request(
    request: &mut HttpRequest,
    credentials: &AwsCredentials,
    region: &str,
    service: &str,
) -> Result<()> {
    sign_request_at(request, credentials, region, service, SystemTime::now())
}

/// [`sign_request`] as of `now`, so tests can pin the timestamp.
fn sign_request_at(
    request: &mut HttpRequest,
    credentials: &AwsCredentials,
    region: &str,
    service: &str,
    now: SystemTime,
) -> Result<()> {
    for name in &GENERATED {
        request.headers.remove(name);
    }

    // The signer derives `host` from the URL but leaves the wire header to the
    // transport. Setting it here keeps what is sent identical to what is signed.
    let host = request
        .url
        .host_str()
        .ok_or_else(|| Error::configuration("aws requests need a host to sign"))?;
    let host = match request.url.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_owned(),
    };
    request.headers.insert(header::HOST, header_value(&host)?);

    let identity = credentials.clone().into();

    let params = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name(service)
        .time(now)
        .settings(SigningSettings::default())
        .build()
        .map_err(|error| {
            Error::configuration("aws: signing parameters are incomplete").with_source(error)
        })?
        .into();

    // The signer borrows the request, so its headers are collected before any
    // are written back.
    let signed = {
        let headers: Vec<(&str, &str)> = request
            .headers
            .iter()
            .filter_map(|(name, value)| Some((name.as_str(), value.to_str().ok()?)))
            .collect();
        let signable = SignableRequest::new(
            request.method.as_str(),
            request.url.as_str(),
            headers.iter().copied(),
            SignableBody::Bytes(request.body.as_deref().unwrap_or_default()),
        )
        // The URL carries the caller's model ID, so an unsignable one is a bad
        // request rather than a misconfigured provider.
        .map_err(|error| {
            Error::invalid_request("aws: request cannot be signed").with_source(error)
        })?;

        let (instructions, _signature) = sign(signable, &params)
            .map_err(|error| Error::configuration("aws: signing failed").with_source(error))?
            .into_parts();
        let (headers, query) = instructions.into_parts();
        // Settings put the signature in headers. Were that ever to change, the
        // request would otherwise go out unsigned with nothing to show for it.
        debug_assert!(
            query.is_empty(),
            "signer returned query parameters; the signature would be dropped"
        );
        headers
    };

    for signed in signed {
        let name = HeaderName::try_from(signed.name()).map_err(|error| {
            Error::configuration("aws: signer produced an invalid header name").with_source(error)
        })?;
        let mut value = header_value(signed.value())?;
        // The signer marks the session token sensitive but not the signature,
        // and both carry credentials.
        value.set_sensitive(signed.sensitive() || GENERATED.contains(&name));
        request.headers.insert(name, value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use aws_smithy_types::DateTime;
    use aws_smithy_types::date_time::Format;
    use url::Url;

    use super::*;
    use crate::transport::{HeaderMap, HeaderValue, Method};

    fn at(timestamp: &str) -> SystemTime {
        DateTime::from_str(timestamp, Format::DateTime)
            .expect("test timestamp should be valid")
            .try_into()
            .expect("test timestamp should fit SystemTime")
    }

    /// The AWS SigV4 test-suite credentials.
    fn suite_credentials() -> AwsCredentials {
        AwsCredentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            None,
            "test-suite",
        )
    }

    /// [`suite_credentials`] as a temporary session with a token.
    fn session_credentials() -> AwsCredentials {
        AwsCredentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            Some("session-token".into()),
            None,
            "test-suite",
        )
    }

    /// `get-vanilla` from the AWS SigV4 test suite.
    #[test]
    fn signs_the_get_vanilla_vector() {
        let mut request = HttpRequest {
            method: Method::GET,
            url: Url::parse("https://example.amazonaws.com/").unwrap(),
            headers: HeaderMap::new(),
            body: None,
        };

        sign_request_at(
            &mut request,
            &suite_credentials(),
            "us-east-1",
            "service",
            at("2015-08-30T12:36:00Z"),
        )
        .unwrap();

        assert_eq!(
            request.headers[header::AUTHORIZATION].to_str().unwrap(),
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=host;x-amz-date, \
             Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        );
        assert_eq!(request.headers[AMZ_DATE], "20150830T123600Z");
    }

    /// `post-x-www-form-urlencoded` from the AWS SigV4 test suite.
    #[test]
    fn signs_the_post_form_vector() {
        let mut request = HttpRequest {
            method: Method::POST,
            url: Url::parse("https://example.amazonaws.com/").unwrap(),
            headers: HeaderMap::from_iter([(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/x-www-form-urlencoded"),
            )]),
            body: Some(bytes::Bytes::from_static(b"Param1=value1")),
        };

        sign_request_at(
            &mut request,
            &suite_credentials(),
            "us-east-1",
            "service",
            at("2015-08-30T12:36:00Z"),
        )
        .unwrap();

        assert_eq!(
            request.headers[header::AUTHORIZATION].to_str().unwrap(),
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=content-type;host;x-amz-date, \
             Signature=ff11897932ad3f4e8b18135d722051e5ac45fc38421b1da7b9d196a0fe09473a"
        );
    }

    #[test]
    fn session_tokens_are_signed_and_sent() {
        let credentials = session_credentials();
        let mut request = HttpRequest {
            method: Method::POST,
            url: Url::parse("https://bedrock-runtime.eu-west-1.amazonaws.com/model/a%3Ab/invoke")
                .unwrap(),
            headers: HeaderMap::new(),
            body: Some(bytes::Bytes::from_static(b"{}")),
        };

        sign_request_at(
            &mut request,
            &credentials,
            "eu-west-1",
            "bedrock",
            at("2026-01-02T03:04:05Z"),
        )
        .unwrap();

        let authorization = request.headers[header::AUTHORIZATION].to_str().unwrap();
        assert!(
            authorization.contains("SignedHeaders=host;x-amz-date;x-amz-security-token,"),
            "{authorization}"
        );
        assert_eq!(request.headers[SECURITY_TOKEN], "session-token");
        assert!(request.headers[SECURITY_TOKEN].is_sensitive());
        assert!(request.headers[header::AUTHORIZATION].is_sensitive());
    }

    /// Bedrock model IDs contain `:`, which the canonical path encodes twice
    /// (`%3A` becomes `%253A`). Getting it wrong mismatches every real model.
    /// AWS publishes no vector with an escaped path, so the expected value is
    /// pinned from a third implementation written against the spec.
    #[test]
    fn model_ids_double_encode_their_colon() {
        let mut request = HttpRequest {
            method: Method::POST,
            url: Url::parse(
                "https://bedrock-runtime.eu-west-1.amazonaws.com\
                 /model/anthropic.claude-sonnet-4%3A0/invoke",
            )
            .unwrap(),
            headers: HeaderMap::new(),
            body: Some(bytes::Bytes::from_static(b"{}")),
        };

        sign_request_at(
            &mut request,
            &suite_credentials(),
            "eu-west-1",
            "bedrock",
            at("2026-01-02T03:04:05Z"),
        )
        .unwrap();

        assert_eq!(
            request.headers[header::AUTHORIZATION].to_str().unwrap(),
            "AWS4-HMAC-SHA256 \
             Credential=AKIDEXAMPLE/20260102/eu-west-1/bedrock/aws4_request, \
             SignedHeaders=host;x-amz-date, \
             Signature=8afd2b6963ee767baf970c6cf7c33e9280b4af6811cd0c7bed0719d90d516b88"
        );
    }

    #[test]
    fn signing_again_replaces_the_previous_signature() {
        let mut request = HttpRequest {
            method: Method::GET,
            url: Url::parse("https://example.amazonaws.com/").unwrap(),
            headers: HeaderMap::new(),
            body: None,
        };
        let credentials = session_credentials();
        sign_request_at(
            &mut request,
            &credentials,
            "us-east-1",
            "service",
            at("2015-08-30T12:36:00Z"),
        )
        .unwrap();
        let first = request.headers[header::AUTHORIZATION].clone();

        sign_request_at(
            &mut request,
            &credentials,
            "us-east-1",
            "service",
            at("2015-08-30T12:37:00Z"),
        )
        .unwrap();

        assert_ne!(request.headers[header::AUTHORIZATION], first);
        for name in &GENERATED {
            assert_eq!(
                request.headers.get_all(name).iter().count(),
                1,
                "{name} must not accumulate"
            );
        }
        assert_eq!(request.headers[AMZ_DATE], "20150830T123700Z");
    }
}
