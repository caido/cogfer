use cogfer::transport::mock::MockTransport;
use cogfer::{
    AssistantPart, Compaction, CompactionPart, ErrorKind, FinishReason, Message, ProviderMetadata,
    ProviderToolPart, ReasoningConfig, ReasoningContent, ReasoningEffort, ReasoningOutput,
    ReasoningPart, Request, StreamEvent, StructuredOutput, ToolCall, ToolResultPart,
};
use serde_json::json;

use crate::common::*;

fn minimal_completed() -> serde_json::Value {
    json!({
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "model": "gpt-5.6",
        "output": [
            {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "ok", "annotations": []}]}
        ],
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                   "input_tokens_details": {"cached_tokens": 0}}
    })
}

mod compaction;
mod request;
mod response;
mod streaming;
