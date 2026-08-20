//! Recording side of cassettes: an [`HttpTransport`] wrapper that copies every
//! exchange as it passes through and writes them with
//! [`RecordingTransport::save`].
//!
//! Credentials never reach the file: request header values outside a small
//! allowlist are stored as `<redacted>`, form-encoded bodies are dropped, the
//! values of [`SCRUBBED_JSON_KEYS`] are replaced in every body (JSON or SSE
//! text), and response headers are limited to the ones the SDK reads.
//! Everything else in a body is stored as sent, so review a fresh cassette
//! before committing it. Authenticators keep their own transport, so token
//! exchanges do not normally pass through here. The form-body and token-key
//! redaction is defense in depth.

use std::ops::Range;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use caido_ai::transport::{HttpByteStream, HttpRequest, HttpResponse, HttpTransport};
use caido_ai::{ApiProfile, Result};
use futures_util::StreamExt;

use super::cassette::{
    Body, Cassette, Chunk, Exchange, Headers, RecordedRequest, RecordedResponse,
    SCRUBBED_JSON_KEYS, hex_encode, scrub_json,
};

impl Body {
    fn from_bytes(bytes: &[u8]) -> Self {
        if let Ok(mut value) = serde_json::from_slice(bytes) {
            scrub_json(&mut value);
            return Body::Json(value);
        }
        let scrubbed = scrub_bytes(bytes, &unscrubbed_value_ranges(bytes));
        match String::from_utf8(scrubbed) {
            Ok(text) => Body::Text(text),
            Err(error) => Body::Hex(hex_encode(error.as_bytes())),
        }
    }
}

impl Chunk {
    fn from_bytes(bytes: &[u8]) -> Self {
        match std::str::from_utf8(bytes) {
            Ok(text) => Chunk::Text(text.to_string()),
            Err(_) => Chunk::Bytes {
                hex: hex_encode(bytes),
            },
        }
    }
}

/// An exchange whose response body may still be streaming.
struct Recording {
    request: RecordedRequest,
    status: u16,
    headers: Headers,
    body: RecordingBody,
}

enum RecordingBody {
    Buffered(Bytes),
    /// Filled while the caller drains the stream.
    Streamed(Arc<Mutex<Vec<Bytes>>>),
}

impl std::fmt::Debug for Recording {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recording")
            .field("url", &self.request.url)
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}

/// An [`HttpTransport`] that forwards to `inner` and remembers every exchange.
#[derive(Debug)]
pub(crate) struct RecordingTransport {
    inner: Arc<dyn HttpTransport>,
    recordings: Mutex<Vec<Recording>>,
}

impl RecordingTransport {
    pub(crate) fn new(inner: Arc<dyn HttpTransport>) -> Self {
        Self {
            inner,
            recordings: Mutex::new(Vec::new()),
        }
    }

    /// Write every exchange recorded so far to the cassette for `scenario`,
    /// then forget them so the next scenario starts empty.
    ///
    /// Call this only after every stream returned by the transport has been
    /// drained, because chunks arriving later are lost.
    pub(crate) fn save(&self, profile: ApiProfile, model: &str, scenario: &str) {
        let recordings = std::mem::take(&mut *self.lock());
        let cassette = Cassette {
            profile: profile.as_str().to_string(),
            model: model.to_string(),
            exchanges: recordings
                .into_iter()
                .map(|recording| Exchange {
                    request: recording.request,
                    response: RecordedResponse {
                        status: recording.status,
                        headers: recording.headers,
                        body: match recording.body {
                            RecordingBody::Buffered(bytes) => Body::from_bytes(&bytes),
                            RecordingBody::Streamed(chunks) => Body::Chunks(scrub_chunks(
                                &chunks
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner),
                            )),
                        },
                    },
                })
                .collect(),
        };
        assert!(
            !cassette.exchanges.is_empty(),
            "scenario `{scenario}` performed no requests; nothing to record"
        );
        // Replay has no authenticator, so a recorded 401-then-retry pair
        // (an expired token refreshed mid-scenario) would never replay.
        assert!(
            cassette
                .exchanges
                .iter()
                .all(|exchange| exchange.response.status != 401),
            "scenario `{scenario}` recorded a 401; refresh the credentials and re-record"
        );
        assert_scrubbed(&cassette, scenario);

        let path = Cassette::path(profile, scenario);
        std::fs::create_dir_all(path.parent().expect("cassette path has a parent"))
            .expect("cassette directory is writable");
        let mut json = serde_json::to_vec_pretty(&cassette).expect("cassette serializes");
        json.push(b'\n');
        std::fs::write(&path, json)
            .unwrap_or_else(|error| panic!("failed to write cassette {}: {error}", path.display()));
        eprintln!(
            "recorded {} exchange(s) to {}",
            cassette.exchanges.len(),
            path.display()
        );
    }

    /// Forget every exchange recorded so far without writing a cassette.
    pub(crate) fn discard(&self) {
        self.lock().clear();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Recording>> {
        self.recordings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// The request with header values redacted unless the name is known to be
/// harmless, and form-encoded (token exchange) bodies dropped.
fn record_request(request: &HttpRequest) -> RecordedRequest {
    let mut url = request.url.clone();
    // Our URLs never carry userinfo, but a cassette must not either.
    let _ = url.set_username("");
    let _ = url.set_password(None);
    let form_encoded = request.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("content-type")
            && value.starts_with("application/x-www-form-urlencoded")
    });
    RecordedRequest {
        url: url.to_string(),
        headers: request
            .headers
            .iter()
            .map(|(name, value)| {
                let visible = [
                    "accept",
                    "anthropic-beta",
                    "anthropic-version",
                    "content-type",
                    "http-referer",
                    "openai-beta",
                    "originator",
                    "user-agent",
                    "x-title",
                ]
                .iter()
                .any(|safe| name.eq_ignore_ascii_case(safe));
                let value = if visible { value.as_str() } else { "<redacted>" };
                (name.clone(), value.to_string())
            })
            .collect(),
        body: request.body.as_ref().map(|body| {
            if form_encoded { Body::Redacted } else { Body::from_bytes(body) }
        }),
    }
}

/// Only the response headers the SDK reads plus `date`, which doubles as the
/// recording timestamp. Rate-limit and tracing headers change on every
/// recording and are dropped.
fn record_response_headers(headers: &[(String, String)]) -> Headers {
    headers
        .iter()
        .filter(|(name, _)| {
            [
                "content-type",
                "date",
                "openai-request-id",
                "request-id",
                "retry-after",
                "x-goog-request-id",
                "x-oai-request-id",
                "x-request-id",
                "x-should-retry",
            ]
            .iter()
            .any(|kept| name.eq_ignore_ascii_case(kept))
        })
        .cloned()
        .collect()
}

/// Byte ranges of the values of every `"<key>": "<value>"` pair in `text`
/// whose key is in [`SCRUBBED_JSON_KEYS`] and whose value is not already
/// `<redacted>`, in order. Works on serialized JSON embedded in SSE frames,
/// where re-parsing would lose the exact bytes.
fn unscrubbed_value_ranges(text: &[u8]) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    for key in SCRUBBED_JSON_KEYS {
        let needle = format!("\"{key}\"");
        let mut from = 0;
        while let Some(offset) = find(&text[from..], needle.as_bytes()) {
            let after_key = from + offset + needle.len();
            from = after_key;
            let mut cursor = after_key;
            while text.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                cursor += 1;
            }
            if text.get(cursor) != Some(&b':') {
                continue;
            }
            cursor += 1;
            while text.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                cursor += 1;
            }
            if text.get(cursor) != Some(&b'"') {
                continue;
            }
            let start = cursor + 1;
            // Identifiers and tokens never contain quotes or escapes, so stop
            // at the first backslash rather than guess.
            let Some(length) = text[start..]
                .iter()
                .position(|byte| matches!(byte, b'"' | b'\\'))
            else {
                continue;
            };
            if text.get(start + length) == Some(&b'"')
                && &text[start..start + length] != b"<redacted>"
            {
                ranges.push(start..start + length);
            }
        }
    }
    ranges.sort_by_key(|range| range.start);
    ranges
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// `bytes` with every value in `ranges` replaced by `<redacted>`.
fn scrub_bytes(bytes: &[u8], ranges: &[Range<usize>]) -> Vec<u8> {
    let mut scrubbed = Vec::with_capacity(bytes.len());
    let mut cursor = 0;
    for range in ranges {
        scrubbed.extend_from_slice(&bytes[cursor..range.start]);
        scrubbed.extend_from_slice(b"<redacted>");
        cursor = range.end;
    }
    scrubbed.extend_from_slice(&bytes[cursor..]);
    scrubbed
}

/// Streamed chunks with scrubbed values. Values are matched across the joined
/// body (a chunk boundary can fall inside one), and every boundary is carried
/// to its position in the scrubbed body so the replayed chunking stays real.
fn scrub_chunks(chunks: &[Bytes]) -> Vec<Chunk> {
    let joined = chunks.concat();
    let ranges = unscrubbed_value_ranges(&joined);
    if ranges.is_empty() {
        return chunks
            .iter()
            .map(|chunk| Chunk::from_bytes(chunk))
            .collect();
    }
    let scrubbed = scrub_bytes(&joined, &ranges);
    // Where an original offset lands in the scrubbed body: shifted by the
    // replacements before it, or inside `<redacted>` when it fell inside a
    // replaced value.
    let mapped = |original: usize| -> usize {
        let mut shift = 0isize;
        for range in &ranges {
            if original >= range.end {
                shift += "<redacted>".len() as isize - range.len() as isize;
            } else if original > range.start {
                let into = (original - range.start).min("<redacted>".len());
                return (range.start as isize + shift) as usize + into;
            } else {
                break;
            }
        }
        (original as isize + shift) as usize
    };
    let mut start = 0;
    let mut end = 0;
    chunks
        .iter()
        .map(|chunk| {
            end += chunk.len();
            let chunk = Chunk::from_bytes(&scrubbed[mapped(start)..mapped(end)]);
            start = end;
            chunk
        })
        .collect()
}

/// Panic if any scrubbed key still carries a value anywhere in `cassette`.
/// Redaction happens by construction above. This keeps that true through
/// future edits.
fn assert_scrubbed(cassette: &Cassette, scenario: &str) {
    for exchange in &cassette.exchanges {
        let bodies = exchange
            .request
            .body
            .iter()
            .chain([&exchange.response.body]);
        for body in bodies {
            let text = match body {
                Body::Json(value) => value.to_string().into_bytes(),
                Body::Text(text) => text.clone().into_bytes(),
                Body::Chunks(chunks) => chunks
                    .iter()
                    .flat_map(|chunk| match chunk {
                        Chunk::Text(text) => text.as_bytes().to_vec(),
                        Chunk::Bytes { .. } => Vec::new(),
                    })
                    .collect(),
                Body::Hex(_) | Body::Redacted => continue,
            };
            assert!(
                unscrubbed_value_ranges(&text).is_empty(),
                "scenario `{scenario}` still carries a value for one of {SCRUBBED_JSON_KEYS:?}"
            );
        }
    }
}

#[async_trait::async_trait]
impl HttpTransport for RecordingTransport {
    async fn execute(&self, request: HttpRequest) -> Result<HttpResponse> {
        let recorded = record_request(&request);
        let response = self.inner.execute(request).await?;
        self.lock().push(Recording {
            request: recorded,
            status: response.status,
            headers: record_response_headers(&response.headers),
            body: RecordingBody::Buffered(response.body.clone()),
        });
        Ok(response)
    }

    async fn stream(&self, request: HttpRequest) -> Result<HttpByteStream> {
        let recorded = record_request(&request);
        let response = self.inner.stream(request).await?;
        let chunks = Arc::new(Mutex::new(Vec::new()));
        self.lock().push(Recording {
            request: recorded,
            status: response.status,
            headers: record_response_headers(&response.headers),
            body: RecordingBody::Streamed(chunks.clone()),
        });
        let bytes = response.bytes.inspect(move |item| {
            if let Ok(chunk) = item {
                chunks
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(chunk.clone());
            }
        });
        Ok(HttpByteStream {
            status: response.status,
            headers: response.headers,
            bytes: Box::pin(bytes),
        })
    }
}

mod tests {
    use super::*;

    /// A value split across two chunks is scrubbed as one, and the boundary
    /// stays inside the replaced span so the chunk count is unchanged.
    #[test]
    fn scrubs_values_split_across_chunks() {
        let chunks = [
            Bytes::from_static(br#"data: {"safety_identifier":"user-AB"#),
            Bytes::from_static(b"CDEF\",\"x\":1}\n\n"),
        ];
        let scrubbed = scrub_chunks(&chunks);
        let texts: Vec<&str> = scrubbed
            .iter()
            .map(|chunk| match chunk {
                Chunk::Text(text) => text.as_str(),
                Chunk::Bytes { .. } => panic!("valid UTF-8 stays text"),
            })
            .collect();
        assert_eq!(
            texts.concat(),
            "data: {\"safety_identifier\":\"<redacted>\",\"x\":1}\n\n"
        );
        assert_eq!(texts.len(), 2);
        assert!(texts[0].ends_with("\"<redact"), "{texts:?}");
    }

    #[test]
    fn scrubs_json_bodies_and_leaves_other_keys() {
        let body = Body::from_bytes(br#"{"user_id":"u1","nested":{"email":"a@b","ok":"keep"}}"#);
        let Body::Json(value) = body else {
            panic!("JSON body")
        };
        assert_eq!(value["user_id"], "<redacted>");
        assert_eq!(value["nested"]["email"], "<redacted>");
        assert_eq!(value["nested"]["ok"], "keep");
    }
}
