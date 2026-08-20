//! The injectable HTTP transport boundary.
//!
//! API profiles build neutral [`HttpRequest`]s and a [`HttpTransport`]
//! implementation executes them. Provider code therefore stays independent of
//! concrete HTTP clients and host applications own proxy, TLS, timeout, and
//! connection-pool policy.

pub mod mock;
pub(crate) mod sse;

#[cfg(feature = "reqwest-transport")]
mod reqwest_transport;
use std::fmt;

use bytes::Bytes;
use futures_util::stream::BoxStream;
#[cfg(feature = "reqwest-transport")]
pub use reqwest_transport::{ReqwestTransport, install_default_crypto_provider};
use url::Url;

use crate::error::Result;
use crate::http::redact_headers;

/// A neutral HTTP POST request.
#[derive(Clone)]
pub struct HttpRequest {
    /// Fully resolved endpoint URL.
    pub url: Url,
    /// Ordered header list. Names are matched case-insensitively.
    pub headers: Vec<(String, String)>,
    /// Encoded request body. Current API profiles send JSON or OAuth form data.
    pub body: Option<Bytes>,
}

impl HttpRequest {
    /// Create a POST request with a JSON body.
    ///
    /// # Errors
    ///
    /// Returns an error when `body` cannot be serialized as JSON.
    pub fn post_json(url: Url, body: &impl serde::Serialize) -> Result<Self> {
        let body = serde_json::to_vec(body).map_err(|e| {
            crate::Error::invalid_request(format!("failed to serialize request body: {e}"))
        })?;
        Ok(Self {
            url,
            headers: vec![("content-type".into(), "application/json".into())],
            body: Some(Bytes::from(body)),
        })
    }

    /// Set a header, replacing any existing values with the same name.
    pub fn set_header(&mut self, name: &str, value: impl Into<String>) {
        self.headers
            .retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
        self.headers.push((name.to_string(), value.into()));
    }

    pub fn has_header(&self, name: &str) -> bool {
        self.headers
            .iter()
            .any(|(existing, _)| existing.eq_ignore_ascii_case(name))
    }

    pub(crate) fn bearer_token(&self) -> Option<&str> {
        self.headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
            .map(|(_, value)| value.as_str())
            .and_then(|value| value.strip_prefix("Bearer "))
    }
}

impl fmt::Debug for HttpRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let headers = redact_headers(&self.headers);
        let url = crate::http::sanitized_url(&self.url);
        f.debug_struct("HttpRequest")
            .field("url", &url)
            .field("headers", &headers)
            .field("body_len", &self.body.as_ref().map(Bytes::len))
            .finish()
    }
}

/// A buffered HTTP response.
#[derive(Clone)]
pub struct HttpResponse {
    /// HTTP response status.
    pub status: u16,
    /// Response headers in received order.
    pub headers: Vec<(String, String)>,
    /// Complete buffered response body.
    pub body: Bytes,
}

impl fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpResponse")
            .field("status", &self.status)
            .field("headers", &redact_headers(&self.headers))
            .field("body_len", &self.body.len())
            .finish()
    }
}

/// A streaming HTTP response: status and headers up front, body as a byte stream.
pub struct HttpByteStream {
    /// HTTP response status available before body polling.
    pub status: u16,
    /// Response headers available before body polling.
    pub headers: Vec<(String, String)>,
    /// Incremental response bytes. Transport errors surface as stream items.
    pub bytes: BoxStream<'static, Result<Bytes>>,
}

impl fmt::Debug for HttpByteStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpByteStream")
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}

/// HTTP transport without retries or provider-specific behavior.
///
/// Implementations return every received response, including non-2xx statuses,
/// so protocol decoders can inspect provider error envelopes. Connection,
/// header, and body-read failures surface as [`crate::ErrorKind::Transport`]
/// or [`crate::ErrorKind::Timeout`]. A buffered body over the transport's size
/// limit is [`crate::ErrorKind::MalformedResponse`].
#[async_trait::async_trait]
pub trait HttpTransport: Send + Sync + fmt::Debug {
    /// Execute a request and buffer the entire response body.
    async fn execute(&self, request: HttpRequest) -> Result<HttpResponse>;

    /// Execute a request and return the response body as a byte stream.
    async fn stream(&self, request: HttpRequest) -> Result<HttpByteStream>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_debug_redacts_headers_and_body() {
        let response = HttpResponse {
            status: 200,
            headers: vec![
                ("content-type".into(), "application/json".into()),
                ("set-cookie".into(), "session=secret-cookie".into()),
            ],
            body: Bytes::from_static(br#"{"access_token":"secret-token"}"#),
        };

        let debug = format!("{response:?}");
        assert!(debug.contains("application/json"));
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("secret-cookie"));
        assert!(!debug.contains("secret-token"));
    }
}
