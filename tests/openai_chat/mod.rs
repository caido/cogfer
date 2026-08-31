use llmwire::transport::mock::MockTransport;
use llmwire::{
    ErrorKind, FinishReason, Message, ProviderMetadata, ReasoningConfig, ReasoningEffort, Request,
    StreamEvent, StructuredOutput, ToolCall, ToolResultPart,
};
use serde_json::json;

use crate::common::*;

fn minimal_completion() -> serde_json::Value {
    json!({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-5.6",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": "ok", "refusal": null},
                     "finish_reason": "stop", "logprobs": null}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
    })
}

fn assistant_with(parts: Vec<llmwire::AssistantPart>) -> Message {
    Message::Assistant {
        content: parts,
        provider_metadata: ProviderMetadata::default(),
    }
}

#[tokio::test]
async fn blocking_length_finish_does_not_return_a_tool_call() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "chatcmpl-partial",
            "choices": [{
                "message": {"tool_calls": [{
                    "id": "call_partial",
                    "function": {"name": "delete", "arguments": "{\"path\":\"/tmp\"}"}
                }]},
                "finish_reason": "length"
            }]
        }),
    );
    let result = openai_chat(&mock)
        .language_model("gpt-5.6")
        .generate(tool_request("x"))
        .await
        .expect("length is a result");

    assert_eq!(result.finish.reason, FinishReason::Length);
    assert_eq!(result.tool_calls().count(), 0);
}

#[tokio::test]
async fn blocking_stop_finish_keeps_tool_calls_and_reports_tool_calls() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "chatcmpl-forced",
            "choices": [{
                "message": {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_forced",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}
                }]},
                "finish_reason": "stop"
            }]
        }),
    );
    let mut request = tool_request("Weather in Paris");
    request.tool_choice = Some(llmwire::ToolChoice::Tool {
        name: "get_weather".into(),
    });
    let result = openai_chat(&mock)
        .language_model("gpt-5.6")
        .generate(request)
        .await
        .expect("forced tool call is a result");

    let calls: Vec<&ToolCall> = result.tool_calls().collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].call_id, "call_forced");
    assert_eq!(result.finish.reason, FinishReason::ToolCalls);
    assert_eq!(result.finish.raw.as_deref(), Some("stop"));
}

#[tokio::test]
async fn blocking_refusal_is_a_content_filter_finish() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "chatcmpl-refusal",
            "model": "gpt-5.6-sol",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": null,
                            "refusal": "I can't help with that."},
                "finish_reason": "stop"
            }]
        }),
    );

    let result = openai_chat(&mock)
        .language_model("gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect("refusal is a result");

    assert_eq!(result.finish.reason, FinishReason::ContentFilter);
    assert_eq!(result.text(), "I can't help with that.");
}

mod request;
mod streaming;
mod usage;
