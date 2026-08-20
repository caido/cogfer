use caido_ai::{Message, Request, ToolDefinition};
use serde_json::json;

pub(crate) fn text_request(text: &str) -> Request {
    Request::builder().message(Message::user(text)).build()
}

pub(crate) fn tool_request(text: &str) -> Request {
    Request::builder()
        .message(Message::user(text))
        .tool(ToolDefinition::new(
            "get_weather",
            "Get current weather",
            json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
                "additionalProperties": false
            }),
        ))
        .build()
}
