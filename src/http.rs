//! ApiProfile-neutral HTTP helpers.

use std::time::Duration;

use http::header::{self, HeaderMap, HeaderName, HeaderValue};
use url::Url;

use crate::error::{Error, ErrorKind, Result};

/// Join a relative endpoint path onto a base URL.
pub(crate) fn join_url(base: &Url, path: &str) -> Url {
    let mut url = base.clone();
    let prefix = url.path().trim_end_matches('/');
    url.set_path(&format!("{prefix}/{path}"));
    url
}

/// Render a URL for diagnostics without credentials or opaque query data.
pub(crate) fn sanitized_url(url: &Url) -> String {
    let mut sanitized = url.clone();
    if sanitized.set_username("").is_err() || sanitized.set_password(None).is_err() {
        return format!("{}:<redacted>", url.scheme());
    }
    sanitized.set_query(None);
    sanitized.set_fragment(None);
    sanitized.to_string()
}

/// Response headers carrying a provider request identifier, in lookup order.
static REQUEST_ID_HEADERS: [HeaderName; 5] = [
    HeaderName::from_static("x-request-id"),
    HeaderName::from_static("x-oai-request-id"),
    HeaderName::from_static("openai-request-id"),
    HeaderName::from_static("x-goog-request-id"),
    HeaderName::from_static("request-id"),
];

/// Extract a provider request identifier from response headers.
pub(crate) fn find_request_id(headers: &HeaderMap) -> Option<String> {
    REQUEST_ID_HEADERS
        .iter()
        .find_map(|name| headers.get(name)?.to_str().ok())
        .map(str::to_owned)
}

/// Build a header value from caller-supplied text, rejecting control
/// characters as an invalid request rather than failing inside the transport.
pub(crate) fn header_value(value: &str) -> Result<HeaderValue> {
    HeaderValue::from_str(value)
        .map_err(|_| Error::invalid_request("header value contains invalid characters"))
}

/// A `Bearer` credential value, marked sensitive so transports do not log it.
pub(crate) fn bearer_value(token: &str) -> Result<HeaderValue> {
    let mut value = header_value(&format!("Bearer {token}"))?;
    value.set_sensitive(true);
    Ok(value)
}

/// Headers for `Debug` output and wire traces, with every value outside the
/// allowlist replaced by `<redacted>`.
///
/// This is deliberately an allowlist. [`crate::Credentials::Header`] accepts
/// arbitrary authentication header names, so a denylist of known secrets
/// would leak custom credentials.
pub(crate) fn redact_headers(headers: &HeaderMap) -> Vec<(&str, &str)> {
    static VISIBLE: [HeaderName; 13] = [
        header::ACCEPT,
        header::ACCEPT_ENCODING,
        HeaderName::from_static("anthropic-beta"),
        HeaderName::from_static("anthropic-version"),
        header::CONTENT_LENGTH,
        header::CONTENT_TYPE,
        header::DATE,
        HeaderName::from_static("http-referer"),
        header::RETRY_AFTER,
        header::USER_AGENT,
        HeaderName::from_static("x-openrouter-title"),
        HeaderName::from_static("x-should-retry"),
        HeaderName::from_static("x-title"),
    ];
    headers
        .iter()
        .map(|(name, value)| {
            let visible = VISIBLE.contains(name) || REQUEST_ID_HEADERS.contains(name);
            let value =
                if visible { value.to_str().unwrap_or("<non-ascii>") } else { "<redacted>" };
            (name.as_str(), value)
        })
        .collect()
}

/// Map an HTTP status onto the baseline SDK error kind.
pub(crate) fn error_kind_for_status(status: u16) -> ErrorKind {
    match status {
        400 | 413 | 422 => ErrorKind::InvalidRequest,
        401 => ErrorKind::Authentication,
        402 | 403 => ErrorKind::Permission,
        404 => ErrorKind::NotFound,
        429 => ErrorKind::RateLimited,
        503 | 529 => ErrorKind::Overloaded,
        _ => ErrorKind::Provider,
    }
}

/// Attach shared HTTP metadata without overriding body-derived metadata.
pub(crate) fn enrich_error_from_headers(
    mut error: Error,
    status: u16,
    headers: &HeaderMap,
) -> Error {
    if error.status().is_none() {
        error = error.with_status(status);
    }
    if error.request_id().is_none()
        && let Some(request_id) = find_request_id(headers)
    {
        error = error.with_request_id(request_id);
    }
    if error.retry_after().is_none()
        && let Some(retry_after) = parse_retry_after(headers)
    {
        error = error.with_retry_after(retry_after);
    }
    error
}

/// Parse a `Retry-After` header expressed in seconds.
pub(crate) fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get(header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_url_preserves_prefixes_and_slashes() {
        let base = Url::parse("http://localhost:4000/litellm/v1/").unwrap();
        let joined = join_url(&base, "chat/completions");
        assert_eq!(
            joined.as_str(),
            "http://localhost:4000/litellm/v1/chat/completions"
        );

        let base = Url::parse("https://api.openai.com/v1").unwrap();
        let joined = join_url(&base, "responses");
        assert_eq!(joined.as_str(), "https://api.openai.com/v1/responses");
    }

    #[test]
    fn join_url_preserves_the_base_query_string() {
        let base =
            Url::parse("https://res.openai.azure.com/openai/deployments/d?api-version=2024-06-01")
                .unwrap();
        let joined = join_url(&base, "chat/completions");
        assert_eq!(
            joined.as_str(),
            "https://res.openai.azure.com/openai/deployments/d/chat/completions?api-version=2024-06-01"
        );
    }

    #[test]
    fn request_id_supports_provider_specific_headers() {
        for name in ["openai-request-id", "x-goog-request-id"] {
            let headers = HeaderMap::from_iter([(
                HeaderName::from_static(name),
                HeaderValue::from_static("request-123"),
            )]);
            assert_eq!(find_request_id(&headers).as_deref(), Some("request-123"));
        }
    }

    #[test]
    fn header_redaction_hides_everything_outside_the_allowlist() {
        let headers = HeaderMap::from_iter(
            [
                ("set-cookie", "session=secret-cookie"),
                ("x-api-key", "secret-key"),
                ("x-provider-account", "account-secret"),
                ("X-Request-Id", "request-123"),
                ("content-type", "application/json"),
            ]
            .map(|(name, value)| {
                (
                    HeaderName::from_bytes(name.as_bytes()).unwrap(),
                    HeaderValue::from_static(value),
                )
            }),
        );
        let redacted = redact_headers(&headers);

        assert_eq!(redacted[0].1, "<redacted>");
        assert_eq!(redacted[1].1, "<redacted>");
        assert_eq!(redacted[2].1, "<redacted>");
        assert_eq!(redacted[3].1, "request-123");
        assert_eq!(redacted[4].1, "application/json");
    }
}
