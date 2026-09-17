use cogfer::transport::mock::MockTransport;
use cogfer::transport::{HeaderName, HeaderValue};
use cogfer::{
    AssistantPart, Compaction, ErrorKind, FinishReason, Message, ProviderMetadata, ReasoningConfig,
    ReasoningContent, ReasoningPart, Request, StreamEvent, StructuredOutput, ToolChoice,
    ToolResultPart,
};
use serde_json::json;

use crate::common::*;

fn minimal_message() -> serde_json::Value {
    json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 2}
    })
}

mod compaction;
mod reasoning;
mod request;
mod response;
mod streaming;
