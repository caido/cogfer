use super::*;

#[tokio::test]
async fn blocking_decode() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "resp_2", "object": "response", "status": "completed", "model": "gpt-5.6",
            "output": [
                {"id": "rs_9", "type": "reasoning",
                 "summary": [{"type": "summary_text", "text": "I think"}],
                 "encrypted_content": "BLOB"},
                {"id": "msg_9", "type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "Calling tool", "annotations": []}]},
                {"id": "fc_9", "type": "function_call", "call_id": "call_9",
                 "name": "get_weather", "arguments": "{\"location\":\"Paris\"}",
                 "status": "completed"}
            ],
            "usage": {"input_tokens": 100, "output_tokens": 30, "total_tokens": 130,
                      "input_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 5},
                      "output_tokens_details": {"reasoning_tokens": 12}}
        }),
    );
    let provider = openai_responses(&mock);
    let result = provider
        .language_model("gpt-5.6")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");

    assert_eq!(result.finish.reason, FinishReason::ToolCalls);
    assert_eq!(result.response.id.as_deref(), Some("resp_2"));
    // Fresh input excludes 60 cached and 5 cache-write tokens.
    assert_eq!(result.usage.input_tokens, Some(35));
    assert_eq!(result.usage.cached_input_tokens, Some(60));
    assert_eq!(result.usage.cache_creation_input_tokens, Some(5));
    assert_eq!(result.usage.total_input_tokens(), Some(100));
    assert_eq!(result.usage.reasoning_tokens, Some(12));

    let reasoning: Vec<_> = result.reasoning().collect();
    assert_eq!(reasoning.len(), 1);
    assert_eq!(reasoning[0].id.as_deref(), Some("rs_9"));
    assert!(
        reasoning[0].content.iter().any(
            |content| matches!(content, ReasoningContent::Encrypted { data } if data == "BLOB")
        )
    );

    let calls: Vec<_> = result.tool_calls().collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].call_id, "call_9");
    assert_eq!(calls[0].item_id.as_deref(), Some("fc_9"));
    assert_eq!(result.text(), "Calling tool");

    let replay = result.to_assistant_message();
    let Message::Assistant { content, .. } = &replay else {
        panic!("expected assistant message");
    };
    assert_eq!(content.len(), 3);
}

#[tokio::test]
async fn failed_status_is_error() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({"id": "resp_3", "object": "response", "status": "failed",
                "error": {"code": "server_error", "message": "boom"}, "output": []}),
    );
    let provider = openai_responses(&mock);
    let error = provider
        .language_model("gpt-5.6")
        .generate(text_request("hi"))
        .await
        .expect_err("failed response is an error");
    assert_eq!(error.kind(), ErrorKind::Provider);
    assert_eq!(error.code(), Some("server_error"));
}

#[tokio::test]
async fn failed_content_policy_status_is_typed() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({"id": "resp_policy", "object": "response", "status": "failed",
                "model": "gpt-5.6-sol",
                "error": {"code": "content_policy_violation",
                          "message": "The response was blocked by a safety policy."},
                "output": []}),
    );

    let error = openai_responses(&mock)
        .language_model("gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect_err("policy failure is an error");

    assert_eq!(error.kind(), ErrorKind::ContentPolicy);
    assert_eq!(error.code(), Some("content_policy_violation"));
}

#[tokio::test]
async fn incomplete_maps_to_length() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({"id": "resp_4", "object": "response", "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {"id": "msg", "type": "message", "role": "assistant",
             "status": "incomplete",
             "content": [{"type": "output_text", "text": "part", "annotations": []}]},
            {"id": "fc_partial", "type": "function_call", "call_id": "call_partial",
             "name": "delete", "arguments": "{\"path\":\"/tmp\"}",
             "status": "completed"}
        ]}),
    );
    let provider = openai_responses(&mock);
    let result = provider
        .language_model("gpt-5.6")
        .generate(text_request("hi"))
        .await
        .expect("incomplete is a result, not an error");
    assert_eq!(result.finish.reason, FinishReason::Length);
    assert_eq!(result.finish.raw.as_deref(), Some("max_output_tokens"));
    assert_eq!(result.tool_calls().count(), 0);
}

#[tokio::test]
async fn http_error_decoding() {
    let mock = MockTransport::shared();
    mock.push_response(
        429,
        headers(&[
            ("content-type", "application/json"),
            ("retry-after", "12"),
            ("x-request-id", "req_x"),
        ]),
        serde_json::to_vec(&json!({"error": {"message": "slow down", "type": "rate_limit_error", "param": null, "code": "rate_limit_exceeded"}})).unwrap(),
    );
    let provider = openai_responses(&mock);
    let error = provider
        .language_model("gpt-5.6")
        .generate(text_request("hi"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::RateLimited);
    assert!(error.retryable());
    assert_eq!(
        error.retry_after(),
        Some(std::time::Duration::from_secs(12))
    );
    assert_eq!(error.request_id(), Some("req_x"));

    let mock = MockTransport::shared();
    mock.push_json(
        400,
        &json!({"error": {"message": "too long", "type": "invalid_request_error",
                 "param": "input", "code": "context_length_exceeded"}}),
    );
    let provider = openai_responses(&mock);
    let error = provider
        .language_model("gpt-5.6")
        .generate(text_request("hi"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ContextLength);
    assert!(!error.retryable());
}

#[tokio::test]
async fn assistant_phase_and_refusals_round_trip() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "resp_p", "object": "response", "status": "completed", "model": "gpt-5.6",
            "output": [
                {"id": "msg_p", "type": "message", "role": "assistant", "status": "completed",
                 "phase": "commentary",
                 "content": [{"type": "output_text", "text": "thinking out loud",
                               "annotations": []}]},
                {"id": "msg_r", "type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "refusal", "refusal": "I can't help with that."}]}
            ],
            "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10}
        }),
    );
    let provider = openai_responses(&mock);
    let result = provider
        .language_model("gpt-5.6")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");
    assert_eq!(result.finish.reason, FinishReason::ContentFilter);

    let mock2 = MockTransport::shared();
    mock2.push_json(200, &minimal_completed());
    let provider2 = openai_responses(&mock2);
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(result.to_assistant_message())
        .message(Message::user("go on"))
        .build();
    provider2
        .language_model("gpt-5.6")
        .generate(request)
        .await
        .expect("generate succeeds");

    let input = mock2.request_json(0)["input"].as_array().unwrap().clone();
    let commentary = input
        .iter()
        .find(|item| item["phase"] == "commentary")
        .expect("phase preserved for replay");
    assert_eq!(commentary["content"][0]["type"], "output_text");
    assert!(
        input
            .iter()
            .any(|item| item["content"][0]["type"] == "refusal"),
        "refusal replayed as a refusal part: {input:#?}"
    );
}

#[tokio::test]
async fn queued_response_is_an_error_not_empty_stop() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &serde_json::json!({"id": "resp_1", "status": "in_progress", "output": []}),
    );
    let provider = openai_responses(&mock);
    let error = provider
        .language_model("gpt-5.4")
        .generate(text_request("hi"))
        .await
        .expect_err("in_progress is not a completed generation");
    assert!(error.message().contains("in_progress"), "{error}");
}

#[tokio::test]
async fn insufficient_quota_is_a_permanent_permission_error() {
    let mock = MockTransport::shared();
    mock.push_json(
        429,
        &json!({"error": {"code": "insufficient_quota", "type": "insufficient_quota",
                          "message": "You exceeded your current quota"}}),
    );
    let error = openai_responses(&mock)
        .language_model("gpt-5.6")
        .generate(text_request("hi"))
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Permission);
    assert!(!error.retryable());
}

#[tokio::test]
async fn error_object_on_a_200_response_fails_generation() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({"id": "resp_1", "status": "completed", "output": [],
                "error": {"code": "server_error", "message": "generation failed upstream"}}),
    );
    let error = openai_responses(&mock)
        .language_model("gpt-5.6")
        .generate(text_request("hi"))
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Provider);
    assert_eq!(error.message(), "generation failed upstream");
    assert_eq!(error.code(), Some("server_error"));
}

#[tokio::test]
async fn decoded_turns_record_their_origin_profile() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completed());
    let provider = openai_responses(&mock);

    let result = provider
        .language_model("gpt-5.6")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");

    assert_eq!(
        result.provider_metadata.get("caido-ai").unwrap()["profile"],
        "openai-responses"
    );
    let Message::Assistant {
        provider_metadata, ..
    } = result.to_assistant_message()
    else {
        panic!("assistant message expected");
    };
    assert_eq!(
        provider_metadata.get("caido-ai").unwrap()["profile"],
        "openai-responses"
    );
}
