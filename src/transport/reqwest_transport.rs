//! [`HttpTransport`] backed by `reqwest`.

use std::fmt;
use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;

use super::{HttpByteStream, HttpRequest, HttpResponse, HttpTransport};
use crate::error::{Error, ErrorKind, Result};

/// Install ring as the process-wide Rustls crypto provider unless one is
/// already installed.
///
/// This crate enables reqwest's `rustls-no-provider` feature, so building any
/// reqwest client panics inside reqwest when the process has no provider.
/// [`ReqwestTransport::new`] calls this automatically. Hosts that build their
/// own client for [`ReqwestTransport::from_client`] must install a provider
/// (this one or their own) first. Installation is first-wins.
pub fn install_default_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
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
    /// Build a Rustls client with a 30-second connect timeout, installing ring if needed.
    ///
    /// Ring is installed as the process-wide provider only when no provider is
    /// already installed. Redirects are disabled because reqwest strips
    /// `Authorization` across origins but not provider-specific headers such as
    /// `x-api-key` and `x-goog-api-key`, which could leak credentials and prompts.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be built.
    pub fn new() -> Result<Self> {
        install_default_crypto_provider();
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
    /// Call [`install_default_crypto_provider`] (or install another Rustls
    /// provider) before building `client`. See that function for why. Prefer
    /// a client without a total `timeout` so streams are bounded only by the
    /// idle-read timeout, and disable redirects or restrict them to trusted
    /// same-origin targets.
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
        let HttpRequest { url, headers, body } = request;
        let mut builder = self.client.post(url);
        if !streaming {
            builder = builder.timeout(self.request_timeout);
        }
        for (name, value) in &headers {
            builder = builder.header(name, value);
        }
        if let Some(body) = body {
            builder = builder.body(body);
        }
        builder
    }

    fn map_error(error: reqwest::Error) -> Error {
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
            let chunk = chunk.map_err(Self::map_error)?;
            if chunk.len() > self.max_response_body_bytes.saturating_sub(body.len()) {
                return Err(self.response_body_too_large());
            }
            body.extend_from_slice(&chunk);
        }
        Ok(Bytes::from(body))
    }
}

fn collect_headers(response: &reqwest::Response) -> Vec<(String, String)> {
    response
        .headers()
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_string(),
                String::from_utf8_lossy(value.as_bytes()).into_owned(),
            )
        })
        .collect()
}

#[async_trait::async_trait]
impl HttpTransport for ReqwestTransport {
    async fn execute(&self, request: HttpRequest) -> Result<HttpResponse> {
        let response = self
            .build(request, false)
            .send()
            .await
            .map_err(Self::map_error)?;
        let status = response.status().as_u16();
        let headers = collect_headers(&response);
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
                .map_err(Self::map_error)?;
        let status = response.status().as_u16();
        let headers = collect_headers(&response);
        let idle = self.idle_read_timeout;
        let mut bytes_stream = response.bytes_stream();
        let stream = futures_util::stream::poll_fn(move |cx| bytes_stream.poll_next_unpin(cx));
        let with_timeout = IdleTimeoutStream::new(stream, idle);
        Ok(HttpByteStream {
            status,
            headers,
            bytes: Box::pin(with_timeout),
        })
    }
}

pin_project_lite::pin_project! {
    /// Wraps a byte stream, failing if the gap between items exceeds `idle`.
    struct IdleTimeoutStream<S> {
        #[pin]
        inner: S,
        idle: Duration,
        #[pin]
        sleep: Option<tokio::time::Sleep>,
        timed_out: bool,
    }
}

impl<S> IdleTimeoutStream<S> {
    fn new(inner: S, idle: Duration) -> Self {
        Self {
            inner,
            idle,
            sleep: None,
            timed_out: false,
        }
    }
}

impl<S> futures_util::Stream for IdleTimeoutStream<S>
where
    S: futures_util::Stream<Item = std::result::Result<Bytes, reqwest::Error>>,
{
    type Item = Result<Bytes>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let mut this = self.project();
        if *this.timed_out {
            return std::task::Poll::Ready(None);
        }
        match this.inner.poll_next(cx) {
            std::task::Poll::Ready(item) => {
                this.sleep.set(None);
                std::task::Poll::Ready(
                    item.map(|result| result.map_err(ReqwestTransport::map_error)),
                )
            }
            std::task::Poll::Pending => {
                if this.sleep.as_mut().as_pin_mut().is_none() {
                    this.sleep.set(Some(tokio::time::sleep(*this.idle)));
                }
                match this
                    .sleep
                    .as_mut()
                    .as_pin_mut()
                    .expect("sleep just set")
                    .poll(cx)
                {
                    std::task::Poll::Ready(()) => {
                        *this.timed_out = true;
                        this.sleep.set(None);
                        std::task::Poll::Ready(Some(Err(Error::new(
                            ErrorKind::Timeout,
                            format!("stream idle for longer than {:?} between chunks", this.idle),
                        ))))
                    }
                    std::task::Poll::Pending => std::task::Poll::Pending,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio::io::AsyncWriteExt;

    use super::*;
    use crate::transport::HttpTransport;

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
        let transport = ReqwestTransport::new().unwrap();

        let error = transport
            .execute(HttpRequest {
                url,
                headers: Vec::new(),
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
        let transport = ReqwestTransport::new()
            .unwrap()
            .with_max_response_body_bytes(7);

        let error = transport
            .execute(HttpRequest {
                url,
                headers: Vec::new(),
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
        let transport = ReqwestTransport::new()
            .unwrap()
            .with_max_response_body_bytes(7);

        let error = transport
            .execute(HttpRequest {
                url,
                headers: Vec::new(),
                body: None,
            })
            .await
            .expect_err("decoded body is larger than the limit");

        assert_eq!(error.kind(), ErrorKind::MalformedResponse);
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
        let transport = ReqwestTransport::new()
            .unwrap()
            .with_idle_read_timeout(Duration::from_millis(200));
        let url = url::Url::parse(&format!("http://{addr}/v1")).unwrap();
        let error = transport
            .stream(HttpRequest {
                url,
                headers: Vec::new(),
                body: None,
            })
            .await
            .expect_err("headers never arrive");
        assert_eq!(error.kind(), ErrorKind::Timeout);
    }
}
