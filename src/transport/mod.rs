//! The injectable HTTP transport boundary.
//!
//! API profiles build neutral [`HttpRequest`]s and a [`HttpTransport`]
//! implementation executes them. Provider code therefore stays independent of
//! concrete HTTP clients and host applications own proxy, TLS, timeout, and
//! connection-pool policy.

#[cfg(feature = "aws")]
pub(crate) mod aws_event_stream;
pub(crate) mod framing;
pub mod mock;
pub(crate) mod sse;

#[cfg(feature = "reqwest-transport")]
mod reqwest_transport;
use std::fmt;

use bytes::Bytes;
use futures_util::stream::BoxStream;
pub use http::Method;
pub use http::header::{self, HeaderMap, HeaderName, HeaderValue};
#[cfg(feature = "reqwest-transport")]
pub use reqwest_transport::{ReqwestTransport, install_default_crypto_provider};
use url::Url;

use crate::error::Result;
use crate::http::redact_headers;

/// A neutral HTTP request.
#[derive(Clone)]
pub struct HttpRequest {
    pub method: Method,
    /// Fully resolved endpoint URL.
    pub url: Url,
    pub headers: HeaderMap,
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
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        Ok(Self {
            method: Method::POST,
            url,
            headers,
            body: Some(Bytes::from(body)),
        })
    }

    /// A GET request that accepts a JSON response.
    pub fn get(url: Url) -> Self {
        let mut headers = HeaderMap::new();
        headers.insert(header::ACCEPT, HeaderValue::from_static("application/json"));
        Self {
            method: Method::GET,
            url,
            headers,
            body: None,
        }
    }

    pub(crate) fn bearer_token(&self) -> Option<&str> {
        self.headers
            .get(header::AUTHORIZATION)?
            .to_str()
            .ok()?
            .strip_prefix("Bearer ")
    }
}

impl fmt::Debug for HttpRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let headers = redact_headers(&self.headers);
        let url = crate::http::sanitized_url(&self.url);
        f.debug_struct("HttpRequest")
            .field("method", &self.method)
            .field("url", &url)
            .field("headers", &headers)
            .field("body_len", &self.body.as_ref().map(Bytes::len))
            .finish()
    }
}

/// A buffered HTTP response.
#[derive(Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: HeaderMap,
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
    pub headers: HeaderMap,
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
            headers: HeaderMap::from_iter([
                (
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                ),
                (
                    header::SET_COOKIE,
                    HeaderValue::from_static("session=secret-cookie"),
                ),
            ]),
            body: Bytes::from_static(br#"{"access_token":"secret-token"}"#),
        };

        let debug = format!("{response:?}");
        assert!(debug.contains("application/json"));
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("secret-cookie"));
        assert!(!debug.contains("secret-token"));
    }
}
