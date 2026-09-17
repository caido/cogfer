use super::*;

#[tokio::test]
async fn responses_dialect_targets_api_x_ai() {
    let mock = MockTransport::shared();
    mock.push_json(200, &responses_completed("grok-4.5"));
    let provider = xai(&mock);

    let result = provider
        .language_model("grok-4.5")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");
    assert_eq!(result.text(), "ok");
    assert_eq!(result.finish.reason, FinishReason::Stop);

    let http = &mock.requests()[0];
    assert_eq!(http.url.as_str(), "https://api.x.ai/v1/responses");
    assert_eq!(header(http, "authorization"), Some("Bearer xai-key"));

    let body = mock.request_json(0);
    assert_eq!(body["model"], json!("grok-4.5"));
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
}

#[tokio::test]
async fn chat_dialect_targets_chat_completions() {
    let mock = MockTransport::shared();
    mock.push_json(200, &chat_completed("grok-4.3"));
    let provider = xai_chat(&mock);
    assert_eq!(provider.profile(), ApiProfile::XaiChatCompletions);

    let result = provider
        .language_model("grok-4.3")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");
    assert_eq!(result.text(), "ok");

    let http = &mock.requests()[0];
    assert_eq!(http.url.as_str(), "https://api.x.ai/v1/chat/completions");
    assert_eq!(header(http, "authorization"), Some("Bearer xai-key"));
}

#[tokio::test]
async fn chat_usage_adds_reasoning_to_visible_completion_tokens() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "chatcmpl_usage", "model": "grok-build-0.1",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 215,
                "completion_tokens": 3,
                "total_tokens": 790,
                "completion_tokens_details": {"reasoning_tokens": 572}
            }
        }),
    );
    let provider = provider_with(
        &mock,
        ProviderConfig::new(
            ApiProfile::XaiChatCompletions,
            Credentials::api_key("xai-key"),
        ),
    );
    let result = provider
        .language_model("grok-build-0.1")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");

    assert_eq!(result.usage.output_tokens, Some(575));
    assert_eq!(
        mock.requests()[0].url.as_str(),
        "https://api.x.ai/v1/chat/completions"
    );
}

#[tokio::test]
async fn chat_stream_usage_adds_reasoning_to_visible_completion_tokens() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"chatcmpl_usage","model":"grok-4.3","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl_usage","model":"grok-4.3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        r#"{"id":"chatcmpl_usage","model":"grok-4.3","choices":[],"usage":{"prompt_tokens":194,"completion_tokens":3,"total_tokens":272,"completion_tokens_details":{"reasoning_tokens":75}}}"#,
        "[DONE]",
    ]);
    let result = provider_with(
        &mock,
        ProviderConfig::new(
            ApiProfile::XaiChatCompletions,
            Credentials::api_key("xai-key"),
        ),
    )
    .language_model("grok-4.3")
    .stream(text_request("hi"))
    .await
    .expect("stream establishes")
    .collect_result()
    .await
    .expect("stream completes");

    assert_eq!(result.usage.output_tokens, Some(78));
}

#[tokio::test]
async fn chat_usage_normalizes_without_a_total_token_count() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "chatcmpl_usage", "model": "grok-build-0.1",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "completion_tokens": 3,
                "completion_tokens_details": {"reasoning_tokens": 572}
            }
        }),
    );
    let result = xai_chat(&mock)
        .language_model("grok-build-0.1")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");

    assert_eq!(result.usage.output_tokens, Some(575));
}

#[tokio::test]
async fn chat_usage_saturates_combined_output_tokens() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "chatcmpl_usage", "model": "grok-build-0.1",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "completion_tokens": u64::MAX,
                "completion_tokens_details": {"reasoning_tokens": 1}
            }
        }),
    );
    let result = xai_chat(&mock)
        .language_model("grok-build-0.1")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");

    assert_eq!(result.usage.output_tokens, Some(u64::MAX));
}

#[tokio::test]
async fn responses_dialect_errors_are_attributed_to_xai() {
    let mock = MockTransport::shared();
    mock.push_json(
        429,
        &json!({"error": {"message": "slow down", "code": "rate_limit_exceeded"}}),
    );
    let provider = xai(&mock);
    assert_eq!(provider.profile(), ApiProfile::XaiResponses);

    let error = provider
        .language_model("grok-4.5")
        .generate(text_request("hi"))
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::RateLimited);
    assert_eq!(error.origin(), Some("xai-responses"));
}

#[tokio::test]
async fn foreign_profile_turn_drops_encrypted_reasoning() {
    let mock = MockTransport::shared();
    mock.push_json(200, &responses_completed("grok-4.5"));
    let provider = xai(&mock);
    let request = cogfer::Request::builder()
        .message(cogfer::Message::user("hi"))
        .message(cogfer::Message::Assistant {
            content: vec![cogfer::AssistantPart::Reasoning(cogfer::ReasoningPart {
                id: Some("rs_1".into()),
                content: vec![cogfer::ReasoningContent::Encrypted { data: "ENC".into() }],
                provider_metadata: cogfer::ProviderMetadata::default(),
            })],
            provider_metadata: cogfer::ProviderMetadata::with(
                "cogfer",
                json!({"profile": "openai-responses"}),
            ),
        })
        .message(cogfer::Message::user("next"))
        .build();

    let result = provider
        .language_model("grok-4.5")
        .generate(request)
        .await
        .expect("generate succeeds");

    let input = mock.request_json(0)["input"].clone();
    assert!(
        input
            .as_array()
            .unwrap()
            .iter()
            .all(|item| item["type"] != json!("reasoning")),
        "openai encrypted reasoning must not reach xai: {input}"
    );
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.message.contains("openai-responses")),
        "{:?}",
        result.warnings
    );
    assert_eq!(
        result.provider_metadata.get("cogfer").unwrap()["profile"],
        "xai-responses"
    );
}

#[tokio::test]
async fn flat_envelopes_tell_a_wrong_key_by_its_message() {
    for (status, code, message, kind) in [
        (
            400,
            "invalid-argument",
            "Incorrect API key provided. You can obtain an API key from https://console.x.ai.",
            ErrorKind::Authentication,
        ),
        (
            400,
            "invalid-argument",
            "Model not found: no-such-model",
            ErrorKind::InvalidRequest,
        ),
        (
            401,
            "unauthenticated:no-credentials",
            "No credentials presented.",
            ErrorKind::Authentication,
        ),
        (
            403,
            "unauthenticated:bad-credentials",
            "The OAuth2 access token could not be validated.",
            ErrorKind::Authentication,
        ),
    ] {
        let mock = MockTransport::shared();
        mock.push_json(status, &json!({"code": code, "error": message}));

        let error = xai(&mock).verify().await.unwrap_err();

        assert_eq!(error.kind(), kind, "{code}: {message}");
        assert_eq!(error.message(), message);
        assert_eq!(error.code(), Some(code));
        assert_eq!(error.status(), Some(status));
        assert_eq!(error.origin(), Some("xai-responses"));
    }
}

#[tokio::test]
async fn chat_dialect_decodes_the_flat_envelope_too() {
    let mock = MockTransport::shared();
    mock.push_json(
        400,
        &json!({"code": "invalid-argument", "error": "Incorrect API key provided."}),
    );

    let error = xai_chat(&mock)
        .language_model("grok-4.5")
        .generate(text_request("hi"))
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Authentication);
    assert_eq!(error.origin(), Some("xai-chat"));
}
