//! Incremental protocol-neutral Server-Sent Events framing.
//!
//! Implements the WHATWG framing used by provider streams: `data:` fields,
//! multi-line data, comments, optional spaces after colons, LF/CRLF/CR
//! endings, an optional leading UTF-8 BOM, and a final partial-frame flush.
//! Every provider dispatches on the JSON payload, so `event:` is dropped along
//! with `id:` and `retry:`, and frames without data are omitted.
//!
//! This is hand-rolled because the general-purpose crates do not offer what
//! provider streams need: bounded line and frame buffers, failing on invalid
//! UTF-8 as soon as the line completes rather than at end of stream,
//! dispatching a final frame without its terminating blank line, and errors
//! that never echo response bytes.

use super::framing::{FrameSource, INVALID_UTF8, StreamFrame};

/// Maximum bytes buffered for one unterminated SSE line. Provider frames are
/// much smaller. The limit prevents unbounded growth when line breaks vanish.
pub(crate) const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;

/// Maximum joined `data:` bytes in one frame. A frame can contain many
/// individually valid lines, so this is enforced independently of
/// [`MAX_LINE_BYTES`].
pub(crate) const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Incremental SSE parser that yields complete frames from raw bytes.
#[derive(Debug, Default)]
pub(crate) struct SseParser {
    /// Raw byte buffer for data that has not yet formed a complete line.
    buffer: Vec<u8>,
    /// Accumulated `data:` lines for the frame in progress.
    data: String,
    /// Whether the frame has seen at least one `data:` field. This is
    /// distinct from `data.is_empty()` because an empty `data:` field is a
    /// valid frame payload.
    has_data: bool,
    /// Joined byte length of the accumulated `data:` lines, including the
    /// newlines inserted between them at dispatch.
    data_bytes: usize,
    /// Set when a completed line failed to decode as UTF-8.
    invalid_utf8: bool,
    /// Set when a line or accumulated frame exceeded its configured limit.
    overflowed: bool,
    /// Whether the leading BOM has already been looked for. It is only ever
    /// stripped once, at the start of the stream.
    bom_checked: bool,
}

impl SseParser {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Whether any line so far contained invalid UTF-8.
    pub(crate) fn saw_invalid_utf8(&self) -> bool {
        self.invalid_utf8
    }

    /// Whether a line or accumulated frame exceeded its limit and was discarded.
    pub(crate) fn overflowed(&self) -> bool {
        self.overflowed
    }

    fn process_line(&mut self, line: &str) -> Option<StreamFrame> {
        if line.is_empty() {
            return self.dispatch();
        }
        if line.starts_with(':') {
            // OpenRouter sends comment keepalives while generation is pending.
            return None;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        if field != "data" {
            return None;
        }
        let separator = usize::from(self.has_data);
        let next_bytes = self
            .data_bytes
            .saturating_add(separator)
            .saturating_add(value.len());
        if next_bytes > MAX_FRAME_BYTES {
            self.overflowed = true;
            self.data.clear();
            self.has_data = false;
            self.data_bytes = 0;
            return None;
        }
        self.data_bytes = next_bytes;
        if self.has_data {
            self.data.push('\n');
        }
        self.data.push_str(value);
        self.has_data = true;
        None
    }

    fn dispatch(&mut self) -> Option<StreamFrame> {
        if !self.has_data {
            return None;
        }
        self.data_bytes = 0;
        self.has_data = false;
        let data = std::mem::take(&mut self.data);
        Some(StreamFrame::Data(data))
    }
}

impl FrameSource for SseParser {
    fn push(&mut self, bytes: &[u8]) -> Vec<StreamFrame> {
        if self.invalid_utf8 || self.overflowed {
            return Vec::new();
        }
        self.buffer.extend_from_slice(bytes);
        if !self.bom_checked {
            if self.buffer.len() < 3 && [0xEF, 0xBB, 0xBF].starts_with(self.buffer.as_slice()) {
                return Vec::new();
            }
            if self.buffer.starts_with(&[0xEF, 0xBB, 0xBF]) {
                self.buffer.drain(..3);
            }
            self.bom_checked = true;
        }

        let buffer = std::mem::take(&mut self.buffer);
        let mut frames = Vec::new();
        let mut line_start = 0;
        let mut cursor = 0;
        while cursor < buffer.len() {
            let newline_pos = cursor;
            let next_line = match buffer[cursor] {
                b'\n' => cursor + 1,
                b'\r' if cursor + 1 == buffer.len() => break,
                b'\r' if buffer[cursor + 1] == b'\n' => cursor + 2,
                b'\r' => cursor + 1,
                _ => {
                    cursor += 1;
                    continue;
                }
            };
            let line = &buffer[line_start..newline_pos];
            line_start = next_line;
            cursor = next_line;
            if line.len() > MAX_LINE_BYTES {
                self.overflowed = true;
                break;
            }
            let line = match std::str::from_utf8(line) {
                Ok(line) => line,
                Err(_) => {
                    self.invalid_utf8 = true;
                    break;
                }
            };
            if let Some(frame) = self.process_line(line) {
                frames.push(frame);
            }
            if self.overflowed {
                break;
            }
        }

        if !self.invalid_utf8 && !self.overflowed {
            self.buffer.extend_from_slice(&buffer[line_start..]);
            if self.buffer.len() > MAX_LINE_BYTES {
                self.overflowed = true;
                self.buffer.clear();
            }
        } else {
            self.buffer.clear();
        }
        frames
    }

    /// Flush any partial frame at end of stream (providers occasionally
    /// truncate without the final blank line).
    fn finish(&mut self) -> Option<StreamFrame> {
        if self.invalid_utf8 || self.overflowed {
            return None;
        }
        if !self.buffer.is_empty() {
            let raw = std::mem::take(&mut self.buffer);
            if raw.len() > MAX_LINE_BYTES {
                self.overflowed = true;
                return None;
            }
            let line = match std::str::from_utf8(&raw) {
                Ok(line) => line,
                Err(_) => {
                    self.invalid_utf8 = true;
                    return None;
                }
            };
            let line = line.strip_suffix('\r').unwrap_or(line);
            if let Some(frame) = self.process_line(line) {
                return Some(frame);
            }
        }
        self.dispatch()
    }

    fn corruption(&self) -> Option<&'static str> {
        if self.overflowed {
            Some("provider stream exceeded the maximum SSE frame size")
        } else if self.invalid_utf8 {
            Some(INVALID_UTF8)
        } else {
            None
        }
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
        assert!(parser.push(b"\n\r").is_empty());

        let frames = parser.push(b"\n");

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
    fn large_chunk_of_short_lines_does_not_overflow() {
        let line = b": keepalive\n";
        let repetitions = MAX_LINE_BYTES / line.len() + 1;
        let mut input = Vec::with_capacity(repetitions * line.len());
        for _ in 0..repetitions {
            input.extend_from_slice(line);
        }

        let mut parser = SseParser::new();
        assert!(parser.push(&input).is_empty());
        assert!(!parser.overflowed());
    }

    #[test]
    fn accumulated_frame_data_is_bounded() {
        let mut parser = SseParser::new();
        parser.data_bytes = MAX_FRAME_BYTES;

        assert!(parser.push(b"data: x\n").is_empty());
        assert!(parser.overflowed());
    }

    #[test]
    fn done_sentinel_passes_through() {
        let mut parser = SseParser::new();
        let frames = collect(&mut parser, b"data: [DONE]\n\n");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], StreamFrame::Data("[DONE]".into()));
    }

    #[test]
    fn empty_data_frames_are_not_emitted() {
        let mut parser = SseParser::new();
        let frames = collect(&mut parser, b"event: ping\n\n\n\n");
        assert!(frames.is_empty());
    }
}

#[cfg(test)]
mod utf8_tests {
    use super::*;

    #[test]
    fn multibyte_split_across_chunks_is_reassembled() {
        let mut parser = SseParser::new();
        let payload = "data: {\"t\":\"é\"}\n\n".as_bytes().to_vec();
        let split = payload.iter().position(|&b| b == 0xC3).unwrap() + 1;
        let mut frames = parser.push(&payload[..split]);
        frames.extend(parser.push(&payload[split..]));
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], StreamFrame::Data("{\"t\":\"é\"}".into()));
        assert!(!parser.saw_invalid_utf8());
    }

    #[test]
    fn invalid_utf8_is_flagged() {
        let mut parser = SseParser::new();
        let mut bytes = b"data: {\"t\":\"".to_vec();
        bytes.push(0xFF);
        bytes.extend_from_slice(b"\"}\n\n");
        let frames = parser.push(&bytes);
        assert!(frames.is_empty());
        assert!(parser.saw_invalid_utf8());
    }

    #[test]
    fn valid_frames_before_invalid_utf8_are_preserved() {
        let mut parser = SseParser::new();
        let mut bytes = b"data: {\"t\":\"valid\"}\n\ndata: {\"t\":\"".to_vec();
        bytes.push(0xFF);
        bytes.extend_from_slice(b"\"}\n\n");

        let frames = parser.push(&bytes);

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], StreamFrame::Data("{\"t\":\"valid\"}".into()));
        assert!(parser.saw_invalid_utf8());
    }
}
