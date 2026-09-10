//! The AWS event stream encoding (`application/vnd.amazon.eventstream`).
//!
//! Framing and checksums are [`aws_smithy_eventstream`]. This module feeds it
//! body chunks and maps the messages it returns onto [`StreamFrame`]. Bedrock
//! marks messages with `:message-type`: `event` becomes
//! [`StreamFrame::Data`], while `exception` and `error` become
//! [`StreamFrame::Exception`] so the decoder can fail the stream.

use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};
use aws_smithy_types::event_stream::Message;
use bytes::BytesMut;

use super::framing::{FrameSource, INVALID_UTF8, StreamFrame};

const TARGET: &str = "ai|aws";
const MALFORMED: &str = "provider stream contained a malformed event stream message";

#[derive(Debug, Default)]
pub(crate) struct AwsEventStreamParser {
    decoder: MessageFrameDecoder,
    buffer: BytesMut,
    corruption: Option<&'static str>,
}

impl AwsEventStreamParser {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn fail(&mut self, reason: &'static str) {
        self.corruption = Some(reason);
        self.buffer.clear();
    }
}

impl FrameSource for AwsEventStreamParser {
    fn push(&mut self, bytes: &[u8]) -> Vec<StreamFrame> {
        let mut frames = Vec::new();
        if self.corruption.is_some() {
            return frames;
        }
        self.buffer.extend_from_slice(bytes);
        loop {
            // The decoder keeps the prelude it consumed, so a partial message
            // resumes on the next call rather than restarting.
            match self.decoder.decode_frame(&mut self.buffer) {
                Ok(DecodedFrame::Complete(message)) => match frame_from(&message) {
                    Ok(Some(frame)) => frames.push(frame),
                    Ok(None) => continue,
                    Err(reason) => {
                        self.fail(reason);
                        return frames;
                    }
                },
                Ok(DecodedFrame::Incomplete) => return frames,
                Err(_) => {
                    self.fail(MALFORMED);
                    return frames;
                }
            }
        }
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

/// Map one decoded message onto a frame. Messages without a `:message-type`
/// header are not Bedrock's and are skipped.
fn frame_from(message: &Message) -> Result<Option<StreamFrame>, &'static str> {
    let header = |name: &str| {
        message
            .headers()
            .iter()
            .find(|header| header.name().as_str() == name)
            .and_then(|header| header.value().as_string().ok())
            .map(aws_smithy_types::str_bytes::StrBytes::as_str)
    };
    let payload = || std::str::from_utf8(message.payload()).map_err(|_| INVALID_UTF8);

    let frame = match header(":message-type") {
        Some("event") => StreamFrame::Data(payload()?.to_owned()),
        Some("exception") => StreamFrame::Exception {
            kind: header(":exception-type").unwrap_or("exception").to_owned(),
            payload: payload()?.to_owned(),
        },
        Some("error") => StreamFrame::Exception {
            kind: header(":error-code").unwrap_or("error").to_owned(),
            payload: header(":error-message").unwrap_or_default().to_owned(),
        },
        other => {
            log::warn!(
                target: TARGET,
                "skipping event stream message with :message-type {other:?}"
            );
            return Ok(None);
        }
    };
    Ok(Some(frame))
}

#[cfg(test)]
mod tests {
    use aws_smithy_eventstream::frame::write_message_to;
    use aws_smithy_types::event_stream::{Header, HeaderValue};

    use super::*;
    use crate::transport::mock::encode_message;

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
        let mut parser = AwsEventStreamParser::new();

        assert!(parser.push(head).is_empty());
        let frames = parser.push(tail);

        assert_eq!(
            frames,
            vec![StreamFrame::Data(r#"{"bytes":"e30="}"#.into())]
        );
        assert!(parser.finish().is_none());
        assert_eq!(parser.corruption(), None);
    }

    /// The decoder keeps the prelude once it has read it, so a message
    /// arriving one byte at a time must still decode exactly once.
    #[test]
    fn decodes_an_event_delivered_one_byte_at_a_time() {
        let message = event("{}");
        let mut parser = AwsEventStreamParser::new();
        let mut frames = Vec::new();

        for byte in &message {
            frames.extend(parser.push(&[*byte]));
        }

        assert_eq!(frames, vec![StreamFrame::Data("{}".into())]);
        assert_eq!(parser.corruption(), None);
    }

    #[test]
    fn decodes_two_messages_in_one_chunk() {
        let mut bytes = event("1");
        bytes.extend(event("2"));

        let frames = AwsEventStreamParser::new().push(&bytes);

        assert_eq!(
            frames,
            vec![StreamFrame::Data("1".into()), StreamFrame::Data("2".into())]
        );
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

        let frames = AwsEventStreamParser::new().push(&message);

        assert_eq!(
            frames,
            vec![StreamFrame::Exception {
                kind: "throttlingException".into(),
                payload: r#"{"message":"slow down"}"#.into(),
            }]
        );
    }

    #[test]
    fn non_string_headers_are_skipped() {
        let message = Message::new(&b"{}"[..])
            .add_header(Header::new(":flag", HeaderValue::Bool(true)))
            .add_header(Header::new(":byte", HeaderValue::Byte(7)))
            .add_header(Header::new(":number", HeaderValue::Int64(0)))
            .add_header(Header::new(
                ":message-type",
                HeaderValue::String("event".into()),
            ));
        let mut bytes = Vec::new();
        write_message_to(&message, &mut bytes).expect("test event stream message should be valid");

        let frames = AwsEventStreamParser::new().push(&bytes);

        assert_eq!(frames, vec![StreamFrame::Data("{}".into())]);
    }

    #[test]
    fn a_bad_message_checksum_is_corruption() {
        let mut message = event("{}");
        let last = message.len() - 1;
        message[last] ^= 0xFF;
        let mut parser = AwsEventStreamParser::new();

        assert!(parser.push(&message).is_empty());
        assert_eq!(parser.corruption(), Some(MALFORMED));
        assert!(parser.push(&event("{}")).is_empty(), "stays failed");
    }

    #[test]
    fn a_bad_prelude_checksum_is_corruption() {
        let mut message = event("{}");
        message[8..12].copy_from_slice(&0u32.to_be_bytes());
        let mut parser = AwsEventStreamParser::new();

        assert!(parser.push(&message).is_empty());
        assert_eq!(parser.corruption(), Some(MALFORMED));
    }

    #[test]
    fn frames_before_corruption_are_delivered() {
        let mut bytes = event("1");
        let mut broken = event("2");
        let last = broken.len() - 1;
        broken[last] ^= 0xFF;
        bytes.extend(broken);
        let mut parser = AwsEventStreamParser::new();

        let frames = parser.push(&bytes);

        assert_eq!(frames, vec![StreamFrame::Data("1".into())]);
        assert!(parser.corruption().is_some());
    }

    #[test]
    fn messages_without_a_type_are_skipped() {
        let message = encode_message(&[(":content-type", "application/json")], b"{}");

        assert!(AwsEventStreamParser::new().push(&message).is_empty());
    }

    #[test]
    fn an_invalid_utf8_payload_is_corruption() {
        let message = encode_message(&[(":message-type", "event")], &[0xFF, 0xFE]);

        let mut parser = AwsEventStreamParser::new();
        assert!(parser.push(&message).is_empty());
        assert_eq!(parser.corruption(), Some(INVALID_UTF8));
    }
}
