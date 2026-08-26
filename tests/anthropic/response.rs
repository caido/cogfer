use super::*;

#[tokio::test]
async fn blocking_decode_with_cache_usage() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "msg_2", "type": "message", "role": "assistant", "model": "claude-opus-5",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "SIG2"},
                {"type": "text", "text": "Using the tool."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"location": "Paris"}}
            ],
            "stop_reason": "tool_use", "stop_sequence": null,
            "usage": {"input_tokens": 25, "output_tokens": 89,
                       "cache_creation_input_tokens": 100, "cache_read_input_tokens": 200,
                       "output_tokens_details": {"thinking_tokens": 12}}
        }),
    );
    let provider = anthropic(&mock);
    let result = provider
        .language_model("claude-opus-5")
        .generate(tool_request("weather"))
        .await
        .expect("generate succeeds");

    assert_eq!(result.finish.reason, FinishReason::ToolCalls);
    assert_eq!(result.finish.raw.as_deref(), Some("tool_use"));
    assert_eq!(result.usage.input_tokens, Some(25));
    assert_eq!(result.usage.cached_input_tokens, Some(200));
    assert_eq!(result.usage.cache_creation_input_tokens, Some(100));
    assert_eq!(result.usage.reasoning_tokens, Some(12));
    assert_eq!(result.usage.total_input_tokens(), Some(325));

    let calls: Vec<_> = result.tool_calls().collect();
    assert_eq!(calls[0].call_id, "toolu_1");
    assert_eq!(
        calls[0].parse_arguments::<serde_json::Value>().unwrap(),
        json!({"location": "Paris"})
    );
    let reasoning: Vec<_> = result.reasoning().collect();
    assert!(matches!(
        &reasoning[0].content[0],
        ReasoningContent::Text { signature: Some(signature), .. } if signature == "SIG2"
    ));
}

#[tokio::test]
async fn max_tokens_does_not_return_a_tool_call() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "msg_partial",
            "content": [{"type": "tool_use", "id": "toolu_partial", "name": "delete",
                         "input": {"path": "/tmp"}}],
            "stop_reason": "max_tokens"
        }),
    );
    let result = anthropic(&mock)
        .language_model("claude-opus-5")
        .generate(tool_request("x"))
        .await
        .expect("max tokens is a result");

    assert_eq!(result.finish.reason, FinishReason::Length);
    assert_eq!(result.tool_calls().count(), 0);
}

#[tokio::test]
async fn cyber_refusal_preserves_typed_stop_details() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "msg_refusal",
            "model": "claude-opus-5",
            "content": [],
            "stop_reason": "refusal",
            "stop_details": {
                "type": "refusal",
                "category": "cyber",
                "explanation": "The request could enable cyber harm."
            },
            "usage": {"input_tokens": 12, "output_tokens": 0}
        }),
    );

    let result = anthropic(&mock)
        .language_model("claude-opus-5")
        .generate(text_request("hi"))
        .await
        .expect("refusal is a result");

    assert_eq!(result.finish.reason, FinishReason::ContentFilter);
    assert_eq!(
        result.provider_metadata.get("anthropic"),
        Some(&json!({
            "stop_details": {
                "type": "refusal",
                "category": "cyber",
                "explanation": "The request could enable cyber harm."
            }
        }))
    );
}

#[tokio::test]
async fn error_decoding() {
    let mock = MockTransport::shared();
    mock.push_response(
        429,
        headers(&[("content-type", "application/json"), ("retry-after", "30")]),
        serde_json::to_vec(&json!({"type": "error",
            "error": {"type": "rate_limit_error", "message": "limited"},
            "request_id": "req_a"}))
        .unwrap(),
    );
    let provider = anthropic(&mock);
    let error = provider
        .language_model("claude-opus-5")
        .generate(text_request("hi"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::RateLimited);
    assert_eq!(error.request_id(), Some("req_a"));
    assert_eq!(
        error.retry_after(),
        Some(std::time::Duration::from_secs(30))
    );

    let mock = MockTransport::shared();
    mock.push_json(
        400,
        &json!({"type": "error",
            "error": {"type": "invalid_request_error",
                      "message": "prompt is too long: 250000 tokens > 200000"}}),
    );
    let provider = anthropic(&mock);
    let error = provider
        .language_model("claude-opus-5")
        .generate(text_request("hi"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ContextLength);
}
