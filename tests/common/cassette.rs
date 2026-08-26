//! Cassettes: provider exchanges recorded over the real transport and replayed
//! through [`MockTransport`].
//!
//! One JSON file per scenario lives under `tests/cassettes/<profile>/`. It
//! holds every request the scenario sent (credentials redacted) and every
//! response as received. Streamed bodies keep the exact byte chunks so replay
//! drives the SSE parser with real chunk boundaries. [`Cassette::queue`]
//! pushes the recorded responses onto a [`MockTransport`] in order.
//! Recording lives in the `recorder` module.
//!
//! Bodies are exact except for two edits: JSON bodies are stored parsed (so
//! key order follows `serde_json`, not the wire), and the values of
//! [`SCRUBBED_JSON_KEYS`] are replaced everywhere, including inside SSE text.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use bytes::Bytes;
use caido_ai::ApiProfile;
use caido_ai::transport::mock::MockTransport;
use caido_ai::transport::{HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};

/// A recorded scenario: every exchange the scenario performed, in order.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Cassette {
    /// [`ApiProfile::as_str`] of the recording provider.
    pub profile: String,
    /// The model the scenario targeted.
    pub model: String,
    pub exchanges: Vec<Exchange>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Exchange {
    pub request: RecordedRequest,
    pub response: RecordedResponse,
}

/// Headers keyed by name. Names are unique in every request the SDK builds
/// and in the response allowlist, so a map keeps the file compact.
pub(crate) type Headers = BTreeMap<String, String>;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RecordedRequest {
    pub url: String,
    pub headers: Headers,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<Body>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RecordedResponse {
    pub status: u16,
    pub headers: Headers,
    pub body: Body,
}

/// JSON keys whose values are replaced by `<redacted>` wherever they appear:
/// account identifiers providers echo back (OpenAI `safety_identifier`,
/// OpenRouter `user_id`) and anything credential-shaped.
pub(crate) const SCRUBBED_JSON_KEYS: &[&str] = &[
    "access_token",
    "account_id",
    "api_key",
    "email",
    "id_token",
    "organization_id",
    "refresh_token",
    "safety_identifier",
    "user_id",
];

/// A request or response body in its most readable exact form.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Body {
    /// A JSON body, pretty-printed for review with `serde_json` key order.
    /// Replay re-serializes it compactly, since decoders do not depend on key
    /// order or whitespace.
    Json(serde_json::Value),
    /// A UTF-8 body that is not JSON.
    Text(String),
    /// A body that is neither JSON nor UTF-8, hex encoded.
    Hex(String),
    /// A streamed body as the exact chunks the transport delivered.
    Chunks(Vec<Chunk>),
    /// A body withheld from the recording (form-encoded credentials).
    Redacted,
}

/// One streamed chunk: text when it is valid UTF-8 on its own, hex when a
/// multi-byte character was split across chunk boundaries.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum Chunk {
    Text(String),
    Bytes { hex: String },
}

impl Chunk {
    fn to_bytes(&self) -> Bytes {
        match self {
            Chunk::Text(text) => Bytes::from(text.clone()),
            Chunk::Bytes { hex } => Bytes::from(hex_decode(hex)),
        }
    }
}

impl Cassette {
    /// Path of the cassette for `scenario` under `profile`.
    pub(crate) fn path(profile: ApiProfile, scenario: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("cassettes")
            .join(profile.as_str())
            .join(format!("{scenario}.json"))
    }

    /// Load the cassette for `scenario`, or `None` when it has not been recorded.
    pub(crate) fn load(profile: ApiProfile, scenario: &str) -> Option<Self> {
        let path = Self::path(profile, scenario);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
            Err(error) => panic!("failed to read cassette {}: {error}", path.display()),
        };
        Some(
            serde_json::from_slice(&bytes)
                .unwrap_or_else(|error| panic!("invalid cassette {}: {error}", path.display())),
        )
    }

    /// Queue every recorded response on `mock`, in order.
    pub(crate) fn queue(&self, mock: &MockTransport) {
        for exchange in &self.exchanges {
            let response = &exchange.response;
            let headers: HeaderMap = response
                .headers
                .iter()
                .map(|(name, value)| {
                    (
                        HeaderName::from_bytes(name.as_bytes()).expect("recorded header name"),
                        HeaderValue::from_str(value).expect("recorded header value"),
                    )
                })
                .collect();
            match &response.body {
                Body::Chunks(chunks) => mock.push_stream_chunks(
                    response.status,
                    headers,
                    chunks.iter().map(Chunk::to_bytes).collect(),
                ),
                Body::Json(value) => {
                    mock.push_response(response.status, headers, value.to_string());
                }
                Body::Text(text) => mock.push_response(response.status, headers, text.clone()),
                Body::Hex(hex) => mock.push_response(response.status, headers, hex_decode(hex)),
                Body::Redacted => panic!("cassette response bodies are never redacted"),
            }
        }
    }
}

/// Replace the values of [`SCRUBBED_JSON_KEYS`] anywhere in `value`.
pub(crate) fn scrub_json(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, value) in map {
                if SCRUBBED_JSON_KEYS.contains(&key.as_str()) {
                    *value = serde_json::Value::String("<redacted>".into());
                } else {
                    scrub_json(value);
                }
            }
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(scrub_json),
        _ => {}
    }
}

pub(super) fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hex_decode(hex: &str) -> Vec<u8> {
    assert!(hex.len().is_multiple_of(2), "hex chunk has odd length");
    (0..hex.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&hex[index..index + 2], 16).expect("valid hex chunk"))
        .collect()
}

mod tests {
    use caido_ai::transport::HttpTransport;
    use caido_ai::transport::mock::MockTransport;
    use futures_util::StreamExt;
    use url::Url;

    use super::*;

    /// A multi-byte character split across chunks survives the file format
    /// and replays as the exact bytes.
    #[tokio::test]
    async fn split_utf8_chunks_round_trip_exactly() {
        let bytes = "data: ✅\n\n".as_bytes();
        let (head, tail) = bytes.split_at(7); // splits the 3-byte check mark
        assert!(std::str::from_utf8(head).is_err());
        let cassette = Cassette {
            profile: "openai-responses".into(),
            model: "m".into(),
            exchanges: vec![Exchange {
                request: RecordedRequest {
                    url: "https://example.test/v1/responses".into(),
                    headers: Headers::new(),
                    body: None,
                },
                response: RecordedResponse {
                    status: 200,
                    headers: Headers::from([(
                        "content-type".to_string(),
                        "text/event-stream".to_string(),
                    )]),
                    body: Body::Chunks(vec![
                        Chunk::Bytes {
                            hex: hex_encode(head),
                        },
                        Chunk::Bytes {
                            hex: hex_encode(tail),
                        },
                    ]),
                },
            }],
        };

        let json = serde_json::to_string(&cassette).unwrap();
        assert!(json.contains(r#"{"hex":"64617461"#), "{json}");
        let loaded: Cassette = serde_json::from_str(&json).unwrap();

        let mock = MockTransport::new();
        loaded.queue(&mock);
        let request = caido_ai::transport::HttpRequest {
            method: caido_ai::transport::Method::POST,
            url: Url::parse("https://example.test/v1/responses").unwrap(),
            headers: HeaderMap::new(),
            body: None,
        };
        let stream = mock.stream(request).await.unwrap();
        let chunks: Vec<Bytes> = stream.bytes.map(Result::unwrap).collect().await;
        assert_eq!(chunks, vec![Bytes::from(head), Bytes::from(tail)]);
    }
}
