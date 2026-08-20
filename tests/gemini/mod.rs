use caido_ai::transport::mock::MockTransport;
use caido_ai::{
    AssistantPart, ErrorKind, FinishReason, Message, ProviderMetadata, ReasoningConfig,
    ReasoningEffort, ReasoningOutput, Request, StreamEvent, StructuredOutput, ToolCall,
    ToolResultPart,
};
use serde_json::json;

use crate::common::*;

fn minimal_response() -> serde_json::Value {
    json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]},
                         "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1,
                           "totalTokenCount": 6},
        "modelVersion": "gemini-2.5-flash", "responseId": "r1"
    })
}

mod request;
mod response;
mod streaming;
