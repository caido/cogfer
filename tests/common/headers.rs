//! Header helpers for request assertions and canned responses.

use caido_ai::transport::{HeaderMap, HeaderName, HeaderValue, HttpRequest};

/// The value of `name` on `request`, when present and ASCII.
pub(crate) fn header<'a>(request: &'a HttpRequest, name: &str) -> Option<&'a str> {
    request.headers.get(name)?.to_str().ok()
}

/// Build a header map from name/value pairs.
pub(crate) fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
    pairs
        .iter()
        .map(|(name, value)| {
            (
                HeaderName::from_bytes(name.as_bytes()).expect("valid header name"),
                HeaderValue::from_str(value).expect("valid header value"),
            )
        })
        .collect()
}
