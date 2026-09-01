//! Anthropic models served by Amazon Bedrock's `InvokeModel` operations.
//!
//! Bedrock speaks the Messages API with a few differences: the model is in
//! the URL, `anthropic_version` replaces the version header, streaming uses
//! the AWS event stream encoding with each Anthropic event base64-encoded
//! inside a `chunk`, and errors use the AWS envelopes decoded by
//! [`crate::protocols::bedrock`].

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde::Deserialize;

use super::stream::AnthropicStreamDecoder;
use crate::error::{Error, Result};
use crate::protocols::bedrock::{bedrock_error_kind, exception_message};
use crate::protocols::{ApiProfile, StreamDecoder};
use crate::stream::{StreamEvent, StreamNormalizer};

/// The body field Bedrock requires in place of the `anthropic-version` header.
pub(super) const ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

/// The message for a stream exception. The payload is a JSON envelope for
/// `:message-type: exception` but bare `:error-message` text for
/// `:message-type: error`, so both shapes have to be handled.
fn stream_exception_message(kind: &str, payload: &str) -> String {
    exception_message(payload.as_bytes())
        .or_else(|| {
            Some(payload.trim())
                .filter(|text| !text.is_empty())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| format!("bedrock stream failed with {kind}"))
}

/// The payload of a Bedrock `chunk` event.
#[derive(Deserialize)]
struct PayloadPart {
    bytes: String,
}

/// Unwraps Bedrock's stream envelope around the Anthropic events.
pub(super) struct BedrockStreamDecoder {
    inner: AnthropicStreamDecoder,
}

impl BedrockStreamDecoder {
    pub(super) fn new() -> Self {
        Self {
            inner: AnthropicStreamDecoder::new(ApiProfile::BedrockAnthropic),
        }
    }
}

impl StreamDecoder for BedrockStreamDecoder {
    fn on_frame(
        &mut self,
        data: String,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) -> Result<()> {
        let part: PayloadPart = serde_json::from_str(&data)
            .map_err(|error| Error::malformed(format!("bedrock: invalid stream chunk: {error}")))?;
        let bytes = STANDARD
            .decode(&part.bytes)
            .map_err(|_| Error::malformed("bedrock: stream chunk is not valid base64"))?;
        let event = String::from_utf8(bytes)
            .map_err(|_| Error::malformed("bedrock: stream chunk is not valid UTF-8"))?;
        self.inner.on_frame(event, normalizer, out)
    }

    fn on_exception(
        &mut self,
        kind: String,
        payload: String,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let error_kind = bedrock_error_kind(Some(&kind), 200);
        let message = stream_exception_message(&kind, &payload);
        let error = Error::new(error_kind, message)
            .with_origin(ApiProfile::BedrockAnthropic.as_str())
            .with_code(kind);
        normalizer.fail(out, error);
    }

    fn on_eof(&mut self, normalizer: &mut StreamNormalizer, out: &mut Vec<StreamEvent>) {
        self.inner.on_eof(normalizer, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_error_messages_survive_a_non_json_payload() {
        assert_eq!(
            stream_exception_message("modelStreamErrorException", "the model stopped"),
            "the model stopped"
        );
        assert_eq!(
            stream_exception_message("throttlingException", r#"{"message":"slow down"}"#),
            "slow down"
        );
        assert_eq!(
            stream_exception_message("InternalFailure", "  "),
            "bedrock stream failed with InternalFailure"
        );
    }
}
