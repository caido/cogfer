//! Blocking and streaming decoders must agree.
//!
//! Every profile keeps two decoders: one for a buffered response body and one
//! for the SSE transcript of the same response. These tests feed each pair the
//! same logical response and require the blocking [`GenerateResult`] to equal
//! the one a consumer assembles from the stream, so the two paths cannot drift
//! in content, identities, finish reason, usage, or metadata.

use llmwire::transport::mock::MockTransport;
use llmwire::{
    AssistantPart, Credentials, FinishReason, GenerateResult, ProviderConfig, Request,
    ToolDefinition,
};
use serde_json::{Value, json};

use crate::common::{collect, drain, provider_with, text_request};

fn tool_request() -> Request {
    let mut request = text_request("What time is it in Paris?");
    request.tools.push(ToolDefinition::new(
        "get_time",
        "Current time in a city",
        json!({"type": "object", "properties": {"city": {"type": "string"}}}),
    ));
    request
}

/// Run `request` once buffered and once streamed and return both results.
async fn both_paths(
    config: ProviderConfig,
    model: &str,
    request: Request,
    buffered: &Value,
    frames: &[&str],
) -> (GenerateResult, GenerateResult) {
    let mock = MockTransport::shared();
    mock.push_json(200, buffered);
    let generated = provider_with(&mock, config.clone())
        .language_model(model)
        .generate(request.clone())
        .await
        .expect("blocking generate succeeds");

    let mock = MockTransport::shared();
    mock.push_sse(frames);
    let events = drain(
        provider_with(&mock, config)
            .language_model(model)
            .stream(request)
            .await
            .expect("stream establishes"),
    )
    .await;
    let streamed = collect(&events).expect("stream completes");
    (generated, streamed)
}

/// Guard against both paths agreeing on an empty or degenerate result.
fn assert_tool_turn(result: &GenerateResult, parts: usize) {
    assert_eq!(result.content.len(), parts, "{result:?}");
    assert_eq!(result.text(), "Hi");
    assert!(
        result
            .content
            .iter()
            .any(|part| matches!(part, AssistantPart::ToolCall(call) if call.arguments == r#"{"city":"Paris"}"#)),
        "{result:?}"
    );
    assert_eq!(result.finish.reason, FinishReason::ToolCalls);
    assert!(result.usage.output_tokens.is_some());
    assert!(result.response.id.is_some());
}

#[tokio::test]
async fn anthropic_decoders_agree() {
    let buffered = json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
        "content": [
            {"type": "thinking", "thinking": "Think.", "signature": "SIG"},
            {"type": "text", "text": "Hi"},
            {"type": "tool_use", "id": "toolu_1", "name": "get_time", "input": {"city": "Paris"}}
        ],
        "stop_reason": "tool_use", "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 2}
    });
    let frames = [
        r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":2}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Think."}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hi"}}"#,
        r#"{"type":"content_block_stop","index":1}"#,
        r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_time","input":{}}}"#,
        r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}"#,
        r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}"#,
        r#"{"type":"content_block_stop","index":2}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":10,"output_tokens":20,"cache_read_input_tokens":2}}"#,
        r#"{"type":"message_stop"}"#,
    ];

    let (generated, streamed) = both_paths(
        ProviderConfig::anthropic(Credentials::api_key("k")),
        "claude-x",
        tool_request(),
        &buffered,
        &frames,
    )
    .await;

    assert_tool_turn(&generated, 3);
    assert_eq!(generated, streamed);
}

#[tokio::test]
async fn gemini_decoders_agree() {
    let buffered = json!({
        "candidates": [{
            "content": {"parts": [
                {"text": "Hi", "thoughtSignature": "SIG"},
                {"functionCall": {"name": "get_time", "args": {"city": "Paris"}, "id": "call_1"}}
            ], "role": "model"},
            "finishReason": "STOP", "index": 0
        }],
        "modelVersion": "gemini-x", "responseId": "r1",
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "thoughtsTokenCount": 5, "totalTokenCount": 35}
    });
    let frames = [
        r#"{"candidates":[{"content":{"parts":[{"text":"Hi","thoughtSignature":"SIG"}],"role":"model"},"index":0}],"modelVersion":"gemini-x","responseId":"r1"}"#,
        r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_time","args":{"city":"Paris"},"id":"call_1"}}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"thoughtsTokenCount":5,"totalTokenCount":35},"modelVersion":"gemini-x","responseId":"r1"}"#,
    ];

    let (generated, streamed) = both_paths(
        ProviderConfig::gemini(Credentials::api_key("k")),
        "gemini-x",
        tool_request(),
        &buffered,
        &frames,
    )
    .await;

    assert_tool_turn(&generated, 2);
    assert_eq!(generated, streamed);
}

#[tokio::test]
async fn openai_chat_decoders_agree() {
    let buffered = json!({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-x",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "Hi",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_time", "arguments": "{\"city\":\"Paris\"}"}}]
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
                  "prompt_tokens_details": {"cached_tokens": 2},
                  "completion_tokens_details": {"reasoning_tokens": 5}}
    });
    let frames = [
        r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-x","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-x","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_time","arguments":""}}]},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-x","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-x","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30,"prompt_tokens_details":{"cached_tokens":2},"completion_tokens_details":{"reasoning_tokens":5}}}"#,
        "[DONE]",
    ];

    let (generated, streamed) = both_paths(
        ProviderConfig::openai_chat(Credentials::api_key("k")),
        "gpt-x",
        tool_request(),
        &buffered,
        &frames,
    )
    .await;

    assert_tool_turn(&generated, 2);
    assert_eq!(generated, streamed);
}

#[tokio::test]
async fn openai_responses_decoders_agree() {
    let output = json!([
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Think."}], "encrypted_content": "ENC"},
        {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
         "content": [{"type": "output_text", "text": "Hi", "annotations": []}]},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_time",
         "arguments": "{\"city\":\"Paris\"}", "status": "completed"}
    ]);
    let usage = json!({"input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                       "input_tokens_details": {"cached_tokens": 2},
                       "output_tokens_details": {"reasoning_tokens": 5}});
    let buffered = json!({
        "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-x",
        "output": output, "usage": usage
    });
    let created = json!({"type": "response.created", "response": {"id": "resp_1", "object": "response", "status": "in_progress", "model": "gpt-x", "output": []}, "sequence_number": 0}).to_string();
    let completed = json!({"type": "response.completed", "response": {"id": "resp_1", "object": "response", "status": "completed", "model": "gpt-x", "output": output, "usage": usage}, "sequence_number": 15}).to_string();
    let frames = [
        created.as_str(),
        r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]},"sequence_number":1}"#,
        r#"{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"Think.","sequence_number":2}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Think."}],"encrypted_content":"ENC"},"sequence_number":3}"#,
        r#"{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_1","status":"in_progress","role":"assistant","content":[]},"sequence_number":4}"#,
        r#"{"type":"response.content_part.added","item_id":"msg_1","output_index":1,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]},"sequence_number":5}"#,
        r#"{"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"delta":"Hi","sequence_number":6}"#,
        r#"{"type":"response.content_part.done","item_id":"msg_1","output_index":1,"content_index":0,"part":{"type":"output_text","text":"Hi","annotations":[]},"sequence_number":7}"#,
        r#"{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hi","annotations":[]}]},"sequence_number":8}"#,
        r#"{"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_time","arguments":"","status":"in_progress"},"sequence_number":9}"#,
        r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":2,"delta":"{\"city\":","sequence_number":10}"#,
        r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":2,"delta":"\"Paris\"}","sequence_number":11}"#,
        r#"{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":2,"arguments":"{\"city\":\"Paris\"}","sequence_number":12}"#,
        r#"{"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_time","arguments":"{\"city\":\"Paris\"}","status":"completed"},"sequence_number":13}"#,
        completed.as_str(),
    ];

    let (generated, streamed) = both_paths(
        ProviderConfig::openai_responses(Credentials::api_key("k")),
        "gpt-x",
        tool_request(),
        &buffered,
        &frames,
    )
    .await;

    assert_tool_turn(&generated, 3);
    assert_eq!(generated, streamed);
}
