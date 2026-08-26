//! [`HttpTransport`] backed by `reqwest`.

use std::fmt;
use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;

use super::{HttpByteStream, HttpRequest, HttpResponse, HttpTransport};
use crate::error::{Error, ErrorKind, Result};

/// Install ring as the process-wide Rustls crypto provider unless one is
/// already installed. Returns whether this call installed it.
///
/// This crate enables reqwest's `rustls-no-provider` feature, so building any
/// reqwest client panics inside reqwest when the process has no provider.
/// Selecting a provider is a process-wide decision, so the transport never
/// makes it implicitly: call this (or install another provider) once at
/// startup, before [`ReqwestTransport::new`] or building a client for
/// [`ReqwestTransport::from_client`]. Installation is first-wins.
pub fn install_default_crypto_provider() -> bool {
    rustls::crypto::ring::default_provider()
        .install_default()
        .is_ok()
}

/// [`HttpTransport`] implementation using `reqwest`.
///
/// Preserves client network policy and adds generation timeouts:
///
/// * buffered requests get a total `request_timeout`
/// * streaming requests get **no** total deadline from this transport
///   (streams legitimately run long) but an `idle_read_timeout` bounding both
///   the wait for response headers and silence between chunks. A client-level
///   `reqwest::ClientBuilder::timeout` on an injected client still applies to
///   streams as a total deadline
/// * dropping a stream aborts the underlying HTTP request.
#[derive(Clone)]
#[must_use = "transport modifiers return an updated value"]
pub struct ReqwestTransport {
    client: reqwest::Client,
    request_timeout: Duration,
    idle_read_timeout: Duration,
    max_response_body_bytes: usize,
}

impl fmt::Debug for ReqwestTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReqwestTransport")
            .field("request_timeout", &self.request_timeout)
            .field("idle_read_timeout", &self.idle_read_timeout)
            .field("max_response_body_bytes", &self.max_response_body_bytes)
            .finish_non_exhaustive()
    }
}

impl ReqwestTransport {
    /// Build a Rustls client with a 30-second connect timeout.
    ///
    /// Redirects are disabled because reqwest strips `Authorization` across
    /// origins but not provider-specific headers such as `x-api-key` and
    /// `x-goog-api-key`, which could leak credentials and prompts.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Configuration`] when no process-wide Rustls crypto
    /// provider is installed (see [`install_default_crypto_provider`]), or if
    /// the HTTP client cannot be built.
    pub fn new() -> Result<Self> {
        if rustls::crypto::CryptoProvider::get_default().is_none() {
            return Err(Error::configuration(
                "no process-wide rustls crypto provider is installed; call \
                 caido_ai::transport::install_default_crypto_provider() or install \
                 your own before building the transport",
            ));
        }
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| {
                Error::configuration("failed to build reqwest client").with_source(error)
            })?;
        Ok(Self::from_client(client))
    }

    /// Wrap a configured client, preserving its network policy. Buffered
    /// requests time out after 300 s, streams after 120 s of silence, and
    /// buffered bodies are capped at 16 MiB until changed with the builder
    /// methods below.
    ///
    /// Install a Rustls crypto provider before building `client`; see
    /// [`install_default_crypto_provider`] for why. Prefer a client without a
    /// total `timeout` so streams are bounded only by the idle-read timeout,
    /// and disable redirects or restrict them to trusted same-origin targets.
    pub fn from_client(client: reqwest::Client) -> Self {
        Self {
            client,
            request_timeout: Duration::from_secs(300),
            idle_read_timeout: Duration::from_secs(120),
            max_response_body_bytes: 16 * 1024 * 1024,
        }
    }

    /// Total timeout for buffered requests.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Maximum wait for response headers or the next streamed chunk.
    ///
    /// Exceeding the duration fails the stream with [`ErrorKind::Timeout`].
    pub fn with_idle_read_timeout(mut self, timeout: Duration) -> Self {
        self.idle_read_timeout = timeout;
        self
    }

    /// Maximum buffered response body size in bytes.
    ///
    /// Oversized bodies fail with [`ErrorKind::MalformedResponse`] without
    /// buffering the remainder. A limit of zero accepts only empty bodies.
    pub fn with_max_response_body_bytes(mut self, max_bytes: usize) -> Self {
        self.max_response_body_bytes = max_bytes;
        self
    }

    fn build(&self, request: HttpRequest, streaming: bool) -> reqwest::RequestBuilder {
        let HttpRequest {
            method,
            url,
            headers,
            body,
        } = request;
        let mut builder = self.client.request(method, url).headers(headers);
        if !streaming {
            builder = builder.timeout(self.request_timeout);
        }
        if let Some(body) = body {
            builder = builder.body(body);
        }
        builder
    }

    fn response_body_too_large(&self) -> Error {
        Error::new(
            ErrorKind::MalformedResponse,
            format!(
                "buffered response body exceeded the configured {} byte limit",
                self.max_response_body_bytes
            ),
        )
    }

    async fn collect_response_body(&self, response: reqwest::Response) -> Result<Bytes> {
        let content_length = response.content_length();
        if content_length.is_some_and(|length| {
            usize::try_from(length).map_or(true, |length| length > self.max_response_body_bytes)
        }) {
            return Err(self.response_body_too_large());
        }

        // The declared length is untrusted, so grow the buffer as bytes arrive.
        let mut body = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_transport_err()?;
            if chunk.len() > self.max_response_body_bytes.saturating_sub(body.len()) {
                return Err(self.response_body_too_large());
            }
            body.extend_from_slice(&chunk);
        }
        Ok(Bytes::from(body))
    }
}

/// Classify reqwest failures into transport errors that never echo the URL.
trait ReqwestResultExt<T> {
    fn map_transport_err(self) -> Result<T>;
}

impl<T> ReqwestResultExt<T> for std::result::Result<T, reqwest::Error> {
    fn map_transport_err(self) -> Result<T> {
        self.map_err(|error| {
            let kind = if error.is_timeout() {
                ErrorKind::Timeout
            } else if error.is_builder() {
                // Invalid headers and similar caller input are not network errors.
                ErrorKind::InvalidRequest
            } else {
                ErrorKind::Transport
            };
            let error = error.without_url();
            Error::new(kind, error.to_string()).with_source(error)
        })
    }
}

#[async_trait::async_trait]
impl HttpTransport for ReqwestTransport {
    async fn execute(&self, request: HttpRequest) -> Result<HttpResponse> {
        let response = self
            .build(request, false)
            .send()
            .await
            .map_transport_err()?;
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let body = self.collect_response_body(response).await?;
        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }

    async fn stream(&self, request: HttpRequest) -> Result<HttpByteStream> {
        // Apply the idle timeout while waiting for response headers too.
        let response =
            tokio::time::timeout(self.idle_read_timeout, self.build(request, true).send())
                .await
                .map_err(|_| {
                    Error::new(
                        ErrorKind::Timeout,
                        format!("no response headers within {:?}", self.idle_read_timeout),
                    )
                })?
                .map_transport_err()?;
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        Ok(HttpByteStream {
            status,
            headers,
            bytes: Box::pin(idle_timeout_stream(
                response.bytes_stream(),
                self.idle_read_timeout,
            )),
        })
    }
}

/// Forward `bytes`, ending the stream with [`ErrorKind::Timeout`] when the gap
/// between chunks exceeds `idle`. The stream also ends after a transport error.
fn idle_timeout_stream(
    bytes: impl futures_util::Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send,
    idle: Duration,
) -> impl futures_util::Stream<Item = Result<Bytes>> + Send {
    async_stream::stream! {
        let mut bytes = std::pin::pin!(bytes);
        loop {
            match tokio::time::timeout(idle, bytes.next()).await {
                Ok(Some(Ok(chunk))) => yield Ok(chunk),
                Ok(Some(Err(error))) => {
                    yield Err(error).map_transport_err();
                    break;
                }
                Ok(None) => break,
                Err(_) => {
                    yield Err(Error::new(
                        ErrorKind::Timeout,
                        format!("stream idle for longer than {idle:?} between chunks"),
                    ));
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio::io::AsyncWriteExt;

    use super::*;
    use crate::transport::{HeaderMap, HttpTransport, Method};

    async fn serve_once(response: &'static [u8]) -> url::Url {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            socket.write_all(response).await.unwrap();
        });
        url::Url::parse(&format!("http://{addr}/v1")).unwrap()
    }

    #[test]
    fn debug_omits_injected_client_configuration() {
        install_default_crypto_provider();
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_static("Bearer transport-secret"),
        );
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .unwrap();

        let debug = format!("{:?}", ReqwestTransport::from_client(client));

        assert!(
            !debug.contains("transport-secret") && !debug.contains("authorization"),
            "transport debug exposed client configuration: {debug}"
        );
    }

    #[tokio::test]
    async fn execute_removes_url_from_transport_errors() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            drop(socket);
        });
        let url =
            url::Url::parse(&format!("http://{addr}/v1?api_key=transport-query-secret")).unwrap();
        install_default_crypto_provider();
        let transport = ReqwestTransport::new().unwrap();

        let error = transport
            .execute(HttpRequest {
                method: Method::POST,
                url,
                headers: HeaderMap::new(),
                body: None,
            })
            .await
            .expect_err("server closes without a response");
        let source = std::error::Error::source(&error)
            .map(ToString::to_string)
            .unwrap_or_default();

        assert!(
            !error.message().contains("transport-query-secret")
                && !source.contains("transport-query-secret"),
            "transport error exposed request URL: {error}; source: {source}"
        );
    }

    #[tokio::test]
    async fn execute_rejects_declared_body_over_limit() {
        let url = serve_once(
            b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\n12345678",
        )
        .await;
        install_default_crypto_provider();
        let transport = ReqwestTransport::new()
            .unwrap()
            .with_max_response_body_bytes(7);

        let error = transport
            .execute(HttpRequest {
                method: Method::POST,
                url,
                headers: HeaderMap::new(),
                body: None,
            })
            .await
            .expect_err("declared body is larger than the limit");

        assert_eq!(error.kind(), ErrorKind::MalformedResponse);
    }

    #[tokio::test]
    async fn execute_rejects_chunked_body_over_limit() {
        let url = serve_once(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n4\r\n1234\r\n4\r\n5678\r\n0\r\n\r\n",
        )
        .await;
        install_default_crypto_provider();
        let transport = ReqwestTransport::new()
            .unwrap()
            .with_max_response_body_bytes(7);

        let error = transport
            .execute(HttpRequest {
                method: Method::POST,
                url,
                headers: HeaderMap::new(),
                body: None,
            })
            .await
            .expect_err("decoded body is larger than the limit");

        assert_eq!(error.kind(), ErrorKind::MalformedResponse);
    }

    #[tokio::test]
    async fn stream_times_out_between_chunks() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nfirst\r\n")
                .await
                .unwrap();
            // Hold the connection open without sending the next chunk.
            std::future::pending::<()>().await;
        });
        install_default_crypto_provider();
        let transport = ReqwestTransport::new()
            .unwrap()
            .with_idle_read_timeout(Duration::from_millis(200));
        let url = url::Url::parse(&format!("http://{addr}/v1")).unwrap();

        let mut stream = transport
            .stream(HttpRequest {
                method: Method::POST,
                url,
                headers: HeaderMap::new(),
                body: None,
            })
            .await
            .expect("headers arrive");

        let first = stream.bytes.next().await.expect("first chunk").unwrap();
        assert_eq!(first.as_ref(), b"first");
        let error = stream
            .bytes
            .next()
            .await
            .expect("timeout is reported as an item")
            .expect_err("second chunk never arrives");
        assert_eq!(error.kind(), ErrorKind::Timeout);
        assert!(
            stream.bytes.next().await.is_none(),
            "stream ends after the timeout"
        );
    }

    #[tokio::test]
    async fn stream_times_out_waiting_for_response_headers() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut held = Vec::new();
            loop {
                if let Ok((socket, _)) = listener.accept().await {
                    held.push(socket);
                }
            }
        });
        install_default_crypto_provider();
        let transport = ReqwestTransport::new()
            .unwrap()
            .with_idle_read_timeout(Duration::from_millis(200));
        let url = url::Url::parse(&format!("http://{addr}/v1")).unwrap();
        let error = transport
            .stream(HttpRequest {
                method: Method::POST,
                url,
                headers: HeaderMap::new(),
                body: None,
            })
            .await
            .expect_err("headers never arrive");
        assert_eq!(error.kind(), ErrorKind::Timeout);
    }
}
