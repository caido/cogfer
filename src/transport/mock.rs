//! An in-memory [`HttpTransport`] for crate and consumer tests.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use bytes::Bytes;

use super::{
    HeaderMap, HeaderValue, HttpByteStream, HttpRequest, HttpResponse, HttpTransport, header,
};
use crate::error::{Error, ErrorKind, Result};

#[derive(Debug, Clone)]
enum CannedReply {
    Buffered(HttpResponse),
    Stream {
        status: u16,
        headers: HeaderMap,
        chunks: Vec<Bytes>,
        /// When set, the stream yields this transport error after the chunks.
        error_after: Option<ErrorKind>,
    },
}

fn content_type(value: &'static str) -> HeaderMap {
    HeaderMap::from_iter([(header::CONTENT_TYPE, HeaderValue::from_static(value))])
}

/// See the [module docs](self).
#[derive(Debug, Default)]
pub struct MockTransport {
    // Sync locks: every guard is dropped within one statement, before any await.
    replies: Mutex<VecDeque<CannedReply>>,
    requests: Mutex<Vec<HttpRequest>>,
}

/// CRC-32 (IEEE 802.3, as used by gzip and the AWS event stream encoding).
pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Encode one message with string headers.
///
/// Hand-written on purpose: this and the event stream decoder are separate
/// implementations, so a fixture that decodes proves the two agree rather
/// than that one is self-consistent.
pub(crate) fn encode_message(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
    /// Prelude plus both checksums.
    const OVERHEAD: usize = 16;

    let mut header_bytes = Vec::new();
    for (name, value) in headers {
        header_bytes.push(name.len() as u8);
        header_bytes.extend_from_slice(name.as_bytes());
        header_bytes.push(7);
        header_bytes.extend_from_slice(&(value.len() as u16).to_be_bytes());
        header_bytes.extend_from_slice(value.as_bytes());
    }
    let total = (OVERHEAD + header_bytes.len() + payload.len()) as u32;
    let mut message = Vec::with_capacity(total as usize);
    message.extend_from_slice(&total.to_be_bytes());
    message.extend_from_slice(&(header_bytes.len() as u32).to_be_bytes());
    let prelude_crc = crc32(&message);
    message.extend_from_slice(&prelude_crc.to_be_bytes());
    message.extend_from_slice(&header_bytes);
    message.extend_from_slice(payload);
    let message_crc = crc32(&message);
    message.extend_from_slice(&message_crc.to_be_bytes());
    message
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
                headers: content_type("application/json"),
                body: Bytes::from(body.to_string()),
            }));
    }

    /// Queue a raw-body response with headers.
    pub fn push_response(&self, status: u16, headers: HeaderMap, body: impl Into<Bytes>) {
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
                headers: content_type("text/event-stream"),
                chunks,
                error_after: None,
            });
    }

    /// Queue an AWS event stream, one `(event type, payload)` message per
    /// entry and one message per chunk. Bedrock wraps each model event as
    /// `{"bytes": base64}` under the `chunk` event type.
    pub fn push_event_stream(&self, events: &[(&str, &[u8])]) {
        let chunks = events
            .iter()
            .map(|&(event_type, payload)| {
                Bytes::from(encode_message(
                    &[
                        (":event-type", event_type),
                        (":content-type", "application/json"),
                        (":message-type", "event"),
                    ],
                    payload,
                ))
            })
            .collect();
        self.push_stream_chunks(
            200,
            content_type("application/vnd.amazon.eventstream"),
            chunks,
        );
    }

    /// Queue an AWS event stream that ends with an exception message.
    pub fn push_event_stream_then_exception(
        &self,
        events: &[(&str, &[u8])],
        exception_type: &str,
        payload: &[u8],
    ) {
        let mut chunks: Vec<Bytes> = events
            .iter()
            .map(|&(event_type, payload)| {
                Bytes::from(encode_message(
                    &[
                        (":event-type", event_type),
                        (":content-type", "application/json"),
                        (":message-type", "event"),
                    ],
                    payload,
                ))
            })
            .collect();
        chunks.push(Bytes::from(encode_message(
            &[
                (":exception-type", exception_type),
                (":content-type", "application/json"),
                (":message-type", "exception"),
            ],
            payload,
        )));
        self.push_stream_chunks(
            200,
            content_type("application/vnd.amazon.eventstream"),
            chunks,
        );
    }

    /// Queue an AWS event stream that ends with an `error` message.
    ///
    /// Unlike an exception, an error carries no payload: its cause is the
    /// `:error-message` header, as plain text rather than JSON.
    pub fn push_event_stream_then_error(
        &self,
        events: &[(&str, &[u8])],
        error_code: &str,
        error_message: &str,
    ) {
        let mut chunks: Vec<Bytes> = events
            .iter()
            .map(|&(event_type, payload)| {
                Bytes::from(encode_message(
                    &[
                        (":event-type", event_type),
                        (":content-type", "application/json"),
                        (":message-type", "event"),
                    ],
                    payload,
                ))
            })
            .collect();
        chunks.push(Bytes::from(encode_message(
            &[
                (":error-code", error_code),
                (":error-message", error_message),
                (":message-type", "error"),
            ],
            b"",
        )));
        self.push_stream_chunks(
            200,
            content_type("application/vnd.amazon.eventstream"),
            chunks,
        );
    }

    /// Queue a streaming response with exact byte chunks.
    pub fn push_stream_chunks(&self, status: u16, headers: HeaderMap, chunks: Vec<Bytes>) {
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
                headers: content_type("text/event-stream"),
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
