//! ChatGPT subscription backend (`POST {base}/responses`).
//!
//! This is the Responses dialect served at
//! `https://chatgpt.com/backend-api/codex`, authenticated with a ChatGPT OAuth
//! token. Unlike the public Responses API, system messages use top-level
//! `instructions`, sampling parameters and output caps are rejected, and every
//! request streams. Blocking generation decodes the buffered SSE transcript.

use serde_json::Value;

use super::ResponsesDialect;
use super::request::lower_body;
use super::stream::ResponsesStreamDecoder;
use super::types::{ResponseObject, StreamErrorEvent, decode_response_object};
use crate::error::{Error, ErrorKind, Result};
use crate::http::join_url;
use crate::protocols::openai::shared::decode_openai_error;
use crate::protocols::{
    ApiProfile, LoweredRequest, ProtocolContext, ProtocolHandler, StreamDecoder,
};
use crate::response::GenerateResult;
use crate::stream::{StreamAccumulator, StreamNormalizer};
use crate::transport::framing::{FrameSource, StreamFrame};
use crate::transport::sse::SseParser;
use crate::transport::{HeaderMap, HeaderName, HeaderValue, header};
use crate::transport::{HttpRequest, HttpResponse};

pub(crate) struct Handler;

impl ProtocolHandler for Handler {
    fn lower(&self, ctx: &ProtocolContext<'_>, streaming: bool) -> Result<LoweredRequest> {
        let (mut body, warnings) = lower_body(ctx, streaming, ResponsesDialect::ChatGpt)?;
        let object = body.as_object_mut().expect("body is an object");
        let missing_instructions = object
            .get("instructions")
            .and_then(Value::as_str)
            .is_none_or(|instructions| instructions.trim().is_empty());
        // The backend requires instructions. This default only fills a gap.
        if missing_instructions {
            object.insert(
                "instructions".into(),
                Value::from("You are ChatGPT, a helpful AI assistant."),
            );
        }

        let url = join_url(ctx.base_url, "responses");
        let mut http = HttpRequest::post_json(url, &body)?;
        http.headers.insert(
            header::ACCEPT,
            HeaderValue::from_static("text/event-stream"),
        );
        // Identifies the calling application. A provider-level `originator`
        // header overrides it.
        http.headers.insert(
            HeaderName::from_static("originator"),
            HeaderValue::from_static("caido-ai"),
        );
        http.headers.insert(
            header::USER_AGENT,
            HeaderValue::from_static(concat!("caido-ai/", env!("CARGO_PKG_VERSION"))),
        );
        http.headers.insert(
            HeaderName::from_static("session_id"),
            HeaderValue::from_str(&uuid::Uuid::new_v4().to_string())
                .expect("a UUID is a valid header value"),
        );
        Ok(LoweredRequest { http, warnings })
    }

    fn decode_response(
        &self,
        ctx: &ProtocolContext<'_>,
        response: &HttpResponse,
    ) -> Result<GenerateResult> {
        let frames = transcript_frames(&response.body)?;
        let (terminal, stream_error) = scan_terminal(&frames)?;
        match (terminal, stream_error) {
            // Match live streaming: an error event wins even when a successful
            // terminal event also appears in the transcript.
            (_, Some(error)) => Err(error),
            (Some(parsed), None) if !parsed.output.is_empty() => {
                decode_response_object(ctx, parsed)
            }
            // ChatGPT often leaves terminal output empty. Replay incremental
            // events so text, reasoning, and tool calls are not lost.
            (Some(_), None) => reconstruct_from_transcript(frames),
            (None, None) => Err(Error::malformed(
                "chatgpt: response transcript ended without a terminal response event",
            )),
        }
    }

    fn decode_error(&self, status: u16, headers: &HeaderMap, body: &[u8]) -> Error {
        decode_openai_error(ApiProfile::ChatGptResponses, status, headers, body)
    }

    fn new_stream_decoder(&self, ctx: &ProtocolContext<'_>) -> Box<dyn StreamDecoder> {
        Box::new(ResponsesStreamDecoder::new(ctx.profile))
    }
}

/// Parse a buffered SSE transcript without exceeding the line limit.
///
/// Chunking matters because the parser bounds one line, not the total body.
/// A single push of a large transcript could otherwise look like an oversized
/// line.
fn transcript_frames(body: &[u8]) -> Result<Vec<StreamFrame>> {
    let mut parser = SseParser::new();
    let mut frames = Vec::new();
    for chunk in body.chunks(64 * 1024) {
        frames.extend(parser.push(chunk));
    }
    frames.extend(parser.finish());
    if parser.saw_invalid_utf8() {
        return Err(Error::malformed(
            "chatgpt: response transcript contained invalid UTF-8",
        ));
    }
    if parser.overflowed() {
        return Err(Error::malformed(
            "chatgpt: response transcript exceeded the maximum SSE frame size",
        ));
    }
    Ok(frames)
}

/// Find the terminal response object (`response.completed` / `.failed` /
/// `.incomplete`, whose status [`decode_response_object`] translates) and
/// any top-level `error` event in the transcript.
fn scan_terminal(frames: &[StreamFrame]) -> Result<(Option<ResponseObject>, Option<Error>)> {
    let mut stream_error: Option<Error> = None;
    let mut terminal: Option<ResponseObject> = None;
    for frame in frames {
        let data = frame.data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        match value.get("type").and_then(Value::as_str) {
            Some("response.completed" | "response.failed" | "response.incomplete") => {
                let object = value.get("response").cloned().ok_or_else(|| {
                    Error::malformed("chatgpt: terminal event carried no response object")
                })?;
                terminal = Some(serde_json::from_value(object).map_err(|e| {
                    Error::malformed(format!("chatgpt: invalid terminal response object: {e}"))
                })?);
            }
            Some("error") => {
                let event: StreamErrorEvent = serde_json::from_value(value).map_err(|e| {
                    Error::malformed(format!("chatgpt: invalid stream error event: {e}"))
                })?;
                stream_error = Some(event.into_error(ApiProfile::ChatGptResponses));
            }
            _ => {}
        }
    }
    Ok((terminal, stream_error))
}

/// Build a blocking result by replaying the transcript through the stream decoder.
fn reconstruct_from_transcript(frames: Vec<StreamFrame>) -> Result<GenerateResult> {
    let mut decoder = ResponsesStreamDecoder::new(ApiProfile::ChatGptResponses);
    let mut normalizer = StreamNormalizer::new(false);
    let mut events = Vec::new();
    for frame in frames {
        if normalizer.is_finished() {
            break;
        }
        if let Err(error) = decoder.on_frame(frame, &mut normalizer, &mut events) {
            normalizer.fail(&mut events, error);
            break;
        }
    }
    if !normalizer.is_finished() {
        decoder.on_eof(&mut normalizer, &mut events);
        normalizer.on_eof(
            &mut events,
            Error::new(
                ErrorKind::TruncatedStream,
                "stream ended before a terminal event was received",
            )
            .with_origin(ApiProfile::ChatGptResponses.as_str()),
        );
    }

    let mut accumulator = StreamAccumulator::new();
    for event in events {
        accumulator.push(event);
    }
    accumulator.into_result()
}
