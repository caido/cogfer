//! The AWS event stream encoding (`application/vnd.amazon.eventstream`).
//!
//! Each message is a prelude (total length, headers length, prelude CRC),
//! typed headers, a payload, and a CRC over everything before it. Bedrock
//! marks messages with `:message-type` (`event`, `exception`, or `error`) and
//! names them with `:event-type` or `:exception-type`. Events become
//! [`StreamFrame`]s carrying the payload; exceptions and errors become frames
//! with [`StreamFrame::exception`] set so the decoder can fail the stream.

use super::framing::{FrameSource, StreamFrame};
use crate::util::crc32;

/// Maximum accepted message size. Bedrock payloads are far smaller; the limit
/// bounds buffering when the length prelude is corrupt.
const MAX_MESSAGE_BYTES: u32 = 16 * 1024 * 1024;

/// Prelude plus both checksums.
const OVERHEAD: usize = 16;

#[derive(Debug, Default)]
pub(crate) struct EventStreamParser {
    buffer: Vec<u8>,
    corruption: Option<&'static str>,
}

impl EventStreamParser {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn fail(&mut self, reason: &'static str) {
        self.corruption = Some(reason);
        self.buffer.clear();
    }
}

impl FrameSource for EventStreamParser {
    fn push(&mut self, bytes: &[u8]) -> Vec<StreamFrame> {
        let mut frames = Vec::new();
        if self.corruption.is_some() {
            return frames;
        }
        self.buffer.extend_from_slice(bytes);
        let mut offset = 0;
        while self.buffer.len() - offset >= OVERHEAD {
            let message = &self.buffer[offset..];
            let total = u32::from_be_bytes(message[..4].try_into().expect("four bytes"));
            if total > MAX_MESSAGE_BYTES || (total as usize) < OVERHEAD {
                self.fail(
                    "provider stream contained an event stream message with an invalid length",
                );
                return frames;
            }
            if crc32(&message[..8])
                != u32::from_be_bytes(message[8..12].try_into().expect("four bytes"))
            {
                self.fail(
                    "provider stream contained an event stream message with a bad prelude checksum",
                );
                return frames;
            }
            let total = total as usize;
            if message.len() < total {
                break;
            }
            let message = &message[..total];
            match decode_message(message) {
                Ok(frame) => frames.extend(frame),
                Err(reason) => {
                    self.fail(reason);
                    return frames;
                }
            }
            offset += total;
        }
        self.buffer.drain(..offset);
        frames
    }

    fn finish(&mut self) -> Option<StreamFrame> {
        // A partial trailing message is a truncated stream, which the runner
        // reports when no terminal event arrived.
        None
    }

    fn corruption(&self) -> Option<&'static str> {
        self.corruption
    }
}

/// Decode one complete message whose prelude has been verified. Messages
/// without a `:message-type` header are not Bedrock's and are skipped.
fn decode_message(message: &[u8]) -> Result<Option<StreamFrame>, &'static str> {
    let total = message.len();
    let checksum = u32::from_be_bytes(message[total - 4..].try_into().expect("four bytes"));
    if crc32(&message[..total - 4]) != checksum {
        return Err("provider stream contained an event stream message with a bad checksum");
    }
    let headers_len = u32::from_be_bytes(message[4..8].try_into().expect("four bytes")) as usize;
    if headers_len > total - OVERHEAD {
        return Err(
            "provider stream contained an event stream message with an invalid header length",
        );
    }
    let headers = decode_headers(&message[12..12 + headers_len])?;
    let payload = &message[12 + headers_len..total - 4];
    let data = String::from_utf8(payload.to_vec())
        .map_err(|_| "provider stream contained invalid UTF-8; output would be corrupted")?;

    let header = |name: &str| {
        headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    };
    let frame = match header(":message-type") {
        Some("event") => StreamFrame::data(data),
        Some("exception") => StreamFrame {
            data,
            exception: Some(header(":exception-type").unwrap_or("exception").to_owned()),
        },
        Some("error") => StreamFrame {
            data: header(":error-message").unwrap_or_default().to_owned(),
            exception: Some(header(":error-code").unwrap_or("error").to_owned()),
        },
        _ => return Ok(None),
    };
    Ok(Some(frame))
}

/// Parse the header block. Only string values are meaningful here; the other
/// types are skipped by their fixed or prefixed sizes.
fn decode_headers(mut bytes: &[u8]) -> Result<Vec<(String, String)>, &'static str> {
    const MALFORMED: &str =
        "provider stream contained an event stream message with malformed headers";
    let mut headers = Vec::new();
    while !bytes.is_empty() {
        let name_len = bytes[0] as usize;
        bytes = &bytes[1..];
        let name = bytes.get(..name_len).ok_or(MALFORMED)?;
        let name = std::str::from_utf8(name).map_err(|_| MALFORMED)?.to_owned();
        bytes = &bytes[name_len..];
        let value_type = *bytes.first().ok_or(MALFORMED)?;
        bytes = &bytes[1..];
        let value_len = match value_type {
            0 | 1 => 0,
            2 => 1,
            3 => 2,
            4 => 4,
            5 | 8 => 8,
            9 => 16,
            6 | 7 => {
                let len = bytes.get(..2).ok_or(MALFORMED)?;
                bytes = &bytes[2..];
                u16::from_be_bytes(len.try_into().expect("two bytes")) as usize
            }
            _ => return Err(MALFORMED),
        };
        let value = bytes.get(..value_len).ok_or(MALFORMED)?;
        if value_type == 7 {
            let value = std::str::from_utf8(value).map_err(|_| MALFORMED)?;
            headers.push((name, value.to_owned()));
        }
        bytes = &bytes[value_len..];
    }
    Ok(headers)
}

/// Encode one message with string headers, the inverse of [`decode_message`].
pub(crate) fn encode_message(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
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

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn event(payload: &str) -> Vec<u8> {
        encode_message(
            &[
                (":event-type", "chunk"),
                (":content-type", "application/json"),
                (":message-type", "event"),
            ],
            payload.as_bytes(),
        )
    }

    #[test]
    fn decodes_events_split_across_chunks() {
        let message = event(r#"{"bytes":"e30="}"#);
        let (head, tail) = message.split_at(20);
        let mut parser = EventStreamParser::new();

        assert!(parser.push(head).is_empty());
        let frames = parser.push(tail);

        assert_eq!(frames, vec![StreamFrame::data(r#"{"bytes":"e30="}"#)]);
        assert!(parser.finish().is_none());
        assert_eq!(parser.corruption(), None);
    }

    #[test]
    fn decodes_two_messages_in_one_chunk() {
        let mut bytes = event("1");
        bytes.extend(event("2"));

        let frames = EventStreamParser::new().push(&bytes);

        assert_eq!(frames, vec![StreamFrame::data("1"), StreamFrame::data("2")]);
    }

    #[test]
    fn exceptions_carry_their_type() {
        let message = encode_message(
            &[
                (":exception-type", "throttlingException"),
                (":content-type", "application/json"),
                (":message-type", "exception"),
            ],
            br#"{"message":"slow down"}"#,
        );

        let frames = EventStreamParser::new().push(&message);

        assert_eq!(
            frames,
            vec![StreamFrame {
                data: r#"{"message":"slow down"}"#.into(),
                exception: Some("throttlingException".into()),
            }]
        );
    }

    #[test]
    fn non_string_headers_are_skipped() {
        let mut headers = Vec::new();
        for (name, value_type, value) in [
            (":flag", 0u8, &[][..]),
            (":byte", 2, &[7][..]),
            (":stamp", 8, &[0; 8][..]),
        ] {
            headers.push(name.len() as u8);
            headers.extend_from_slice(name.as_bytes());
            headers.push(value_type);
            headers.extend_from_slice(value);
        }
        let name = ":message-type";
        headers.push(name.len() as u8);
        headers.extend_from_slice(name.as_bytes());
        headers.push(7);
        headers.extend_from_slice(&5u16.to_be_bytes());
        headers.extend_from_slice(b"event");
        let total = (OVERHEAD + headers.len() + 2) as u32;
        let mut message = Vec::new();
        message.extend_from_slice(&total.to_be_bytes());
        message.extend_from_slice(&(headers.len() as u32).to_be_bytes());
        message.extend_from_slice(&crc32(&message).to_be_bytes());
        message.extend_from_slice(&headers);
        message.extend_from_slice(b"{}");
        message.extend_from_slice(&crc32(&message).to_be_bytes());

        let frames = EventStreamParser::new().push(&message);

        assert_eq!(frames, vec![StreamFrame::data("{}")]);
    }

    #[test]
    fn a_bad_message_checksum_is_corruption() {
        let mut message = event("{}");
        let last = message.len() - 1;
        message[last] ^= 0xFF;
        let mut parser = EventStreamParser::new();

        assert!(parser.push(&message).is_empty());
        assert!(parser.corruption().unwrap().contains("checksum"));
        assert!(parser.push(&event("{}")).is_empty(), "stays failed");
    }

    #[test]
    fn a_bad_prelude_is_corruption_before_the_body_arrives() {
        let mut message = event("{}");
        message[0..4].copy_from_slice(&(MAX_MESSAGE_BYTES + 1).to_be_bytes());
        let mut parser = EventStreamParser::new();

        assert!(parser.push(&message[..OVERHEAD]).is_empty());
        assert!(parser.corruption().unwrap().contains("length"));
    }

    #[test]
    fn frames_before_corruption_are_delivered() {
        let mut bytes = event("1");
        let mut broken = event("2");
        let last = broken.len() - 1;
        broken[last] ^= 0xFF;
        bytes.extend(broken);
        let mut parser = EventStreamParser::new();

        let frames = parser.push(&bytes);

        assert_eq!(frames, vec![StreamFrame::data("1")]);
        assert!(parser.corruption().is_some());
    }

    #[test]
    fn messages_without_a_type_are_skipped() {
        let message = encode_message(&[(":content-type", "application/json")], b"{}");

        assert!(EventStreamParser::new().push(&message).is_empty());
    }
}
