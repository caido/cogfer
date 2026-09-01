//! Incremental protocol-neutral Server-Sent Events framing.
//!
//! `sse-core` handles arbitrary byte chunks and WHATWG event framing. This
//! adapter exposes only the data payloads consumed by provider protocols.

use std::num::NonZeroUsize;

use sse_core::{SseDecoder, SseEvent};

use super::framing::{FrameSource, StreamFrame};

/// Incremental SSE parser that yields complete data frames from raw bytes.
pub(crate) struct SseParser {
    decoder: SseDecoder,
    decoder_failed: bool,
}

impl SseParser {
    pub(crate) fn new() -> Self {
        Self {
            // Keep framing policy-neutral and avoid rejecting valid large events.
            decoder: SseDecoder::with_limit(NonZeroUsize::MAX),
            decoder_failed: false,
        }
    }

    fn decode(&mut self, bytes: &[u8]) -> Vec<StreamFrame> {
        if self.decoder_failed {
            return Vec::new();
        }

        let mut bytes = bytes;
        let mut frames = Vec::new();
        while let Some(event) = self.decoder.next(&mut bytes) {
            match event {
                Ok(SseEvent::Message(message)) => {
                    frames.push(StreamFrame::Data(message.data));
                }
                Ok(SseEvent::Retry(_)) => {}
                Err(_) => {
                    self.decoder_failed = true;
                    break;
                }
            }
        }
        frames
    }
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameSource for SseParser {
    fn push(&mut self, bytes: &[u8]) -> Vec<StreamFrame> {
        self.decode(bytes)
    }

    /// Flush a provider frame even when its final separator is missing.
    fn finish(&mut self) -> Option<StreamFrame> {
        self.decode(b"\n\n").into_iter().next()
    }

    fn corruption(&self) -> Option<&'static str> {
        self.decoder_failed
            .then_some("provider stream exceeded the SSE decoder capacity")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(parser: &mut SseParser, input: &[u8]) -> Vec<StreamFrame> {
        let mut frames = parser.push(input);
        if let Some(frame) = parser.finish() {
            frames.push(frame);
        }
        frames
    }

    #[test]
    fn parses_simple_data_frames() {
        let mut parser = SseParser::new();
        let frames = collect(&mut parser, b"data: {\"a\":1}\n\ndata: {\"b\":2}\n\n");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], StreamFrame::Data("{\"a\":1}".into()));
        assert_eq!(frames[1], StreamFrame::Data("{\"b\":2}".into()));
    }

    #[test]
    fn ignores_event_field_and_parses_crlf() {
        let mut parser = SseParser::new();
        let frames = collect(
            &mut parser,
            b"event: message_start\r\ndata: {\"type\":\"message_start\"}\r\n\r\n",
        );
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0],
            StreamFrame::Data("{\"type\":\"message_start\"}".into())
        );
    }

    #[test]
    fn parses_lone_carriage_return_line_endings() {
        let mut parser = SseParser::new();
        let frames = collect(&mut parser, b"data: one\r\rdata: two\r\r");

        assert_eq!(
            frames,
            vec![
                StreamFrame::Data("one".into()),
                StreamFrame::Data("two".into()),
            ]
        );
    }

    #[test]
    fn reassembles_crlf_split_across_chunks() {
        let mut parser = SseParser::new();
        assert!(parser.push(b"data: split\r").is_empty());
        assert!(parser.push(b"\n").is_empty());

        let frames = parser.push(b"\r\n");

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], StreamFrame::Data("split".into()));
    }

    #[test]
    fn joins_multi_line_data() {
        let mut parser = SseParser::new();
        let frames = collect(&mut parser, b"data: line1\ndata: line2\n\n");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], StreamFrame::Data("line1\nline2".into()));
    }

    #[test]
    fn skips_comments_and_handles_split_chunks() {
        let mut parser = SseParser::new();
        let mut frames = parser.push(b": OPENROUTER PROCESSING\n\ndata: {\"a\"");
        assert!(frames.is_empty());
        frames.extend(parser.push(b":1}\n"));
        assert!(frames.is_empty());
        frames.extend(parser.push(b"\n"));
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], StreamFrame::Data("{\"a\":1}".into()));
    }

    #[test]
    fn strips_bom_and_no_space_after_colon() {
        let mut parser = SseParser::new();
        let frames = collect(&mut parser, b"\xEF\xBB\xBFdata:{\"a\":1}\n\n");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], StreamFrame::Data("{\"a\":1}".into()));
    }

    #[test]
    fn flushes_truncated_final_frame() {
        let mut parser = SseParser::new();
        let mut frames = parser.push(b"data: {\"a\":1}");
        assert!(frames.is_empty());
        if let Some(frame) = parser.finish() {
            frames.push(frame);
        }
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], StreamFrame::Data("{\"a\":1}".into()));
    }

    #[test]
    fn flushes_frame_dispatched_by_trailing_carriage_return() {
        let mut parser = SseParser::new();
        let frames = collect(&mut parser, b"data: hi\r\n\r");
        assert_eq!(frames, vec![StreamFrame::Data("hi".into())]);
    }

    #[test]
    fn done_sentinel_passes_through() {
        let mut parser = SseParser::new();
        let frames = collect(&mut parser, b"data: [DONE]\n\n");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], StreamFrame::Data("[DONE]".into()));
    }

    #[test]
    fn events_without_data_are_not_emitted() {
        let mut parser = SseParser::new();
        let frames = collect(&mut parser, b"event: ping\n\n\n\n");
        assert!(frames.is_empty());
    }

    #[test]
    fn retry_directives_are_not_emitted() {
        let mut parser = SseParser::new();
        let frames = collect(&mut parser, b"retry: 1000\n\ndata: ready\n\n");
        assert_eq!(frames, vec![StreamFrame::Data("ready".into())]);
    }

    #[test]
    fn multibyte_utf8_split_across_chunks_is_reassembled() {
        let mut parser = SseParser::new();
        let payload = "data: {\"t\":\"é\"}\n\n".as_bytes().to_vec();
        let split = payload.iter().position(|&b| b == 0xC3).unwrap() + 1;
        let mut frames = parser.push(&payload[..split]);
        frames.extend(parser.push(&payload[split..]));
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], StreamFrame::Data("{\"t\":\"é\"}".into()));
    }

    #[test]
    fn invalid_utf8_is_replaced() {
        let mut parser = SseParser::new();
        let mut bytes = b"data: {\"t\":\"".to_vec();
        bytes.push(0xFF);
        bytes.extend_from_slice(b"\"}\n\n");

        let frames = parser.push(&bytes);

        assert_eq!(frames, vec![StreamFrame::Data("{\"t\":\"�\"}".into())]);
        assert_eq!(parser.corruption(), None);
    }
}
