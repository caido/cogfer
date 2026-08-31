//! Header helpers for request assertions and canned responses.

use llmwire::transport::{HeaderMap, HeaderName, HeaderValue, HttpRequest};

pub(crate) fn header<'a>(request: &'a HttpRequest, name: &str) -> Option<&'a str> {
    request.headers.get(name)?.to_str().ok()
}

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
