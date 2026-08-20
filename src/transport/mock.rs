//! An in-memory [`HttpTransport`] for crate and consumer tests.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use bytes::Bytes;

use super::{HttpByteStream, HttpRequest, HttpResponse, HttpTransport};
use crate::error::{Error, ErrorKind, Result};

/// A queued canned reply.
#[derive(Debug, Clone)]
enum CannedReply {
    Buffered(HttpResponse),
    Stream {
        status: u16,
        headers: Vec<(String, String)>,
        chunks: Vec<Bytes>,
        /// When set, the stream yields this transport error after the chunks.
        error_after: Option<ErrorKind>,
    },
}

/// See the [module docs](self).
#[derive(Debug, Default)]
pub struct MockTransport {
    replies: Mutex<VecDeque<CannedReply>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl MockTransport {
    pub fn new() -> Self {
        Self::default()
    }

    /// Shared handle ready for [`crate::Client::builder`].
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Queue a JSON response.
    pub fn push_json(&self, status: u16, body: &serde_json::Value) {
        self.replies
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back(CannedReply::Buffered(HttpResponse {
                status,
                headers: vec![("content-type".into(), "application/json".into())],
                body: Bytes::from(body.to_string()),
            }));
    }

    /// Queue a raw-body response with headers.
    pub fn push_response(
        &self,
        status: u16,
        headers: Vec<(String, String)>,
        body: impl Into<Bytes>,
    ) {
        self.replies
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back(CannedReply::Buffered(HttpResponse {
                status,
                headers,
                body: body.into(),
            }));
    }

    /// Queue an SSE stream delivered as one chunk per `data:` frame.
    ///
    /// Each entry becomes `data: <entry>\n\n`. Prefix an entry with
    /// `"event: name\n"` yourself for named events, or use
    /// [`MockTransport::push_stream_chunks`] for full control.
    pub fn push_sse(&self, frames: &[&str]) {
        let chunks = frames
            .iter()
            .map(|frame| {
                if frame.starts_with("event:") || frame.starts_with(':') {
                    Bytes::from(format!("{frame}\n\n"))
                } else {
                    Bytes::from(format!("data: {frame}\n\n"))
                }
            })
            .collect();
        self.replies
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back(CannedReply::Stream {
                status: 200,
                headers: vec![("content-type".into(), "text/event-stream".into())],
                chunks,
                error_after: None,
            });
    }

    /// Queue a streaming response with exact byte chunks.
    pub fn push_stream_chunks(
        &self,
        status: u16,
        headers: Vec<(String, String)>,
        chunks: Vec<Bytes>,
    ) {
        self.replies
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back(CannedReply::Stream {
                status,
                headers,
                chunks,
                error_after: None,
            });
    }

    /// Queue a stream that fails with a transport error after delivering
    /// `chunks`. A buffered `execute` of this reply fails outright.
    pub fn push_stream_then_error(&self, chunks: Vec<Bytes>, error: ErrorKind) {
        self.replies
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back(CannedReply::Stream {
                status: 200,
                headers: vec![("content-type".into(), "text/event-stream".into())],
                chunks,
                error_after: Some(error),
            });
    }

    /// Every request executed so far, in order.
    pub fn requests(&self) -> Vec<HttpRequest> {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// The JSON body of the `index`-th recorded request.
    ///
    /// # Panics
    ///
    /// Panics when `index` is out of bounds, the request has no body, or the
    /// recorded body is not valid JSON. This helper is intended for tests.
    pub fn request_json(&self, index: usize) -> serde_json::Value {
        let requests = self
            .requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let request = requests.get(index).expect("request index in range");
        serde_json::from_slice(request.body.as_ref().expect("request has a body"))
            .expect("request body is JSON")
    }

    fn next_reply(&self) -> Result<CannedReply> {
        self.replies
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Transport,
                    "MockTransport: no canned reply queued for this request",
                )
            })
    }
}

#[async_trait::async_trait]
impl HttpTransport for MockTransport {
    async fn execute(&self, request: HttpRequest) -> Result<HttpResponse> {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(request);
        match self.next_reply()? {
            CannedReply::Buffered(response) => Ok(response),
            CannedReply::Stream {
                error_after: Some(kind),
                ..
            } => Err(Error::new(kind, "MockTransport: injected transport error")),
            CannedReply::Stream {
                status,
                headers,
                chunks,
                error_after: None,
            } => {
                let mut body = Vec::new();
                for chunk in chunks {
                    body.extend_from_slice(&chunk);
                }
                Ok(HttpResponse {
                    status,
                    headers,
                    body: Bytes::from(body),
                })
            }
        }
    }

    async fn stream(&self, request: HttpRequest) -> Result<HttpByteStream> {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(request);
        match self.next_reply()? {
            CannedReply::Buffered(response) => Ok(HttpByteStream {
                status: response.status,
                headers: response.headers,
                bytes: Box::pin(futures_util::stream::once(async move { Ok(response.body) })),
            }),
            CannedReply::Stream {
                status,
                headers,
                chunks,
                error_after,
            } => {
                let mut items: Vec<Result<Bytes>> = chunks.into_iter().map(Ok).collect();
                if let Some(kind) = error_after {
                    items.push(Err(Error::new(
                        kind,
                        "MockTransport: injected stream error",
                    )));
                }
                Ok(HttpByteStream {
                    status,
                    headers,
                    bytes: Box::pin(futures_util::stream::iter(items)),
                })
            }
        }
    }
}
