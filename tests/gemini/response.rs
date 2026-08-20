use super::*;

#[tokio::test]
async fn blocking_decode_with_thoughts_and_synthetic_ids() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "candidates": [{"content": {"role": "model", "parts": [
                {"text": "Considering the cities...", "thought": true},
                {"functionCall": {"name": "get_weather", "args": {"location": "Paris"}},
                 "thoughtSignature": "SIG_A"},
                {"functionCall": {"name": "get_weather", "args": {"location": "London"}}}
            ]}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 42, "candidatesTokenCount": 31,
                               "thoughtsTokenCount": 256, "totalTokenCount": 329,
                               "cachedContentTokenCount": 10, "toolUsePromptTokenCount": 7},
            "modelVersion": "gemini-2.5-pro", "responseId": "r2"
        }),
    );
    let provider = gemini(&mock);
    let result = provider
        .language_model("gemini-2.5-pro")
        .generate(tool_request("both"))
        .await
        .expect("generate succeeds");

    assert_eq!(result.finish.reason, FinishReason::ToolCalls);
    assert_eq!(result.finish.raw.as_deref(), Some("STOP"));

    assert_eq!(result.usage.output_tokens, Some(31 + 256));
    assert_eq!(result.usage.reasoning_tokens, Some(256));
    assert_eq!(result.usage.cached_input_tokens, Some(10));
    assert_eq!(result.usage.tool_use_prompt_tokens, Some(7));

    let calls: Vec<_> = result.tool_calls().collect();
    assert_eq!(calls.len(), 2);
    assert!(calls[0].call_id.starts_with("call_"));
    assert!(calls[1].call_id.starts_with("call_"));
    assert_ne!(calls[0].call_id, calls[1].call_id);
    assert_eq!(
        calls[0]
            .provider_metadata
            .get("gemini")
            .and_then(|namespace| namespace.get("thought_signature"))
            .and_then(serde_json::Value::as_str),
        Some("SIG_A")
    );
    assert_eq!(result.reasoning().count(), 1);
}

#[tokio::test]
async fn max_tokens_does_not_return_a_function_call() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "candidates": [{
                "content": {"parts": [{"functionCall": {
                    "name": "delete", "args": {"path": "/tmp"}
                }}]},
                "finishReason": "MAX_TOKENS"
            }],
            "responseId": "r_partial"
        }),
    );
    let result = gemini(&mock)
        .language_model("gemini-2.5-flash")
        .generate(tool_request("x"))
        .await
        .expect("max tokens is a result");

    assert_eq!(result.finish.reason, FinishReason::Length);
    assert_eq!(result.tool_calls().count(), 0);
}

#[tokio::test]
async fn blocked_prompt_is_content_filter_not_error() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({"promptFeedback": {"blockReason": "SAFETY", "safetyRatings": []},
                "usageMetadata": {"promptTokenCount": 12, "totalTokenCount": 12}}),
    );
    let provider = gemini(&mock);
    let result = provider
        .language_model("gemini-2.5-flash")
        .generate(text_request("hi"))
        .await
        .expect("blocked prompt is a result");
    assert_eq!(result.finish.reason, FinishReason::ContentFilter);
    assert_eq!(result.finish.raw.as_deref(), Some("SAFETY"));
    assert!(result.content.is_empty());
    assert_eq!(
        result.provider_metadata.get("gemini"),
        Some(&json!({
            "promptFeedback": {"blockReason": "SAFETY", "safetyRatings": []}
        }))
    );
}

#[tokio::test]
async fn filtered_candidate_preserves_safety_details() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "candidates": [{
                "content": {"parts": []},
                "finishReason": "SAFETY",
                "finishMessage": "The response was blocked by a safety filter.",
                "safetyRatings": [{
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "probability": "HIGH",
                    "blocked": true
                }]
            }],
            "modelVersion": "gemini-3.6-flash",
            "responseId": "filtered-1"
        }),
    );

    let result = gemini(&mock)
        .language_model("gemini-3.6-flash")
        .generate(text_request("hi"))
        .await
        .expect("filtered candidate is a result");

    assert_eq!(result.finish.reason, FinishReason::ContentFilter);
    let metadata = result
        .provider_metadata
        .get("gemini")
        .expect("candidate safety metadata");
    assert_eq!(
        metadata["finishMessage"],
        "The response was blocked by a safety filter."
    );
    assert_eq!(metadata["safetyRatings"][0]["blocked"], true);
}

#[tokio::test]
async fn google_status_error_with_retry_info() {
    let mock = MockTransport::shared();
    mock.push_json(
        429,
        &json!({"error": {"code": 429, "message": "quota exceeded",
                 "status": "RESOURCE_EXHAUSTED",
                 "details": [
                   {"@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "37s"}]}}),
    );
    let provider = gemini(&mock);
    let error = provider
        .language_model("gemini-2.5-flash")
        .generate(text_request("hi"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::RateLimited);
    assert_eq!(
        error.retry_after(),
        Some(std::time::Duration::from_secs(37))
    );

    let mock = MockTransport::shared();
    mock.push_json(
        400,
        &json!({"error": {"code": 400, "message": "API key not valid.",
                 "status": "INVALID_ARGUMENT",
                 "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                               "reason": "API_KEY_INVALID", "domain": "googleapis.com"}]}}),
    );
    let provider = gemini(&mock);
    let error = provider
        .language_model("gemini-2.5-flash")
        .generate(text_request("hi"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Authentication);
}
