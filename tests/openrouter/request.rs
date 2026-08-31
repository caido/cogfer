use super::*;

#[tokio::test]
async fn request_uses_openrouter_reasoning_and_sampling() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({"id": "gen-1", "object": "chat.completion", "created": 1,
                "model": "anthropic/claude-sonnet-5",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok",
                             "refusal": null}, "finish_reason": "stop", "logprobs": null}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}),
    );
    let provider = openrouter(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .reasoning(ReasoningConfig::budget(
            std::num::NonZeroU32::new(2048).unwrap(),
        ))
        .top_k(50)
        .max_output_tokens(400)
        .build();
    provider
        .language_model("anthropic/claude-sonnet-5")
        .generate(request)
        .await
        .expect("generate succeeds");

    let http = &mock.requests()[0];
    assert_eq!(
        http.url.as_str(),
        "https://openrouter.ai/api/v1/chat/completions"
    );
    assert_eq!(header(http, "authorization"), Some("Bearer sk-or-test"));
    let body = mock.request_json(0);
    assert_eq!(body["reasoning"]["max_tokens"], 2048);
    assert_eq!(body["top_k"], 50);
    assert_eq!(body["max_tokens"], 400);
    assert!(body.get("max_completion_tokens").is_none());
}

#[tokio::test]
async fn effort_and_output_visibility_are_sent_without_coercion() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "gen-1", "object": "chat.completion", "created": 1, "model": "model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok",
                         "refusal": null}, "finish_reason": "stop", "logprobs": null}]
        }),
    );
    let provider = openrouter(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .reasoning(ReasoningConfig::Effort {
            effort: llmwire::ReasoningEffort::XHigh,
            output: Some(llmwire::ReasoningOutput::Omit),
        })
        .build();

    provider
        .language_model("model")
        .generate(request)
        .await
        .expect("generate succeeds");

    assert_eq!(
        mock.request_json(0)["reasoning"],
        json!({"effort": "xhigh", "exclude": true})
    );
}

#[tokio::test]
async fn disabled_reasoning_is_sent_as_none_effort() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "gen-1", "object": "chat.completion", "created": 1, "model": "model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok",
                         "refusal": null}, "finish_reason": "stop", "logprobs": null}]
        }),
    );
    let provider = openrouter(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .reasoning(ReasoningConfig::Disabled)
        .build();

    provider
        .language_model("model")
        .generate(request)
        .await
        .expect("generate succeeds");

    assert_eq!(mock.request_json(0)["reasoning"], json!({"effort": "none"}));
}

#[tokio::test]
async fn reasoning_details_round_trip() {
    let details = json!([
        {"type": "reasoning.encrypted", "data": "SIGBLOB", "id": null,
         "format": "anthropic-claude-v1", "index": 0},
        {"type": "reasoning.text", "text": "thinking...", "signature": "SIG", "id": null,
         "format": "anthropic-claude-v1", "index": 1}
    ]);
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({"id": "gen-2", "object": "chat.completion", "created": 1,
                "model": "anthropic/claude-sonnet-5",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": "answer",
                                          "refusal": null,
                                          "reasoning": "thinking...",
                                          "reasoning_details": details},
                             "finish_reason": "stop",
                             "native_finish_reason": "end_turn", "logprobs": null}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}),
    );
    let provider = openrouter(&mock);
    let result = provider
        .language_model("anthropic/claude-sonnet-5")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");

    assert_eq!(result.finish.raw.as_deref(), Some("end_turn"));
    let reasoning: Vec<_> = result.reasoning().collect();
    assert_eq!(reasoning.len(), 1);
    let stored = reasoning[0]
        .provider_metadata
        .get("openrouter")
        .and_then(|namespace| namespace.get("reasoning_details"))
        .expect("details preserved");
    assert_eq!(stored, &details);

    let mock2 = MockTransport::shared();
    mock2.push_json(
        200,
        &json!({"id": "gen-3", "object": "chat.completion", "created": 1, "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok",
                             "refusal": null}, "finish_reason": "stop", "logprobs": null}]}),
    );
    let provider2 = openrouter(&mock2);
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(result.to_assistant_message())
        .message(Message::user("next"))
        .build();
    provider2
        .language_model("anthropic/claude-sonnet-5")
        .generate(request)
        .await
        .expect("generate succeeds");
    let body = mock2.request_json(0);
    let assistant = &body["messages"][1];
    assert_eq!(assistant["reasoning_details"], details);
}

#[tokio::test]
async fn foreign_encrypted_reasoning_is_not_rebuilt_into_details() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({"id": "gen-9", "object": "chat.completion", "created": 1, "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok",
                             "refusal": null}, "finish_reason": "stop", "logprobs": null}]}),
    );
    let provider = openrouter(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(Message::Assistant {
            content: vec![
                llmwire::AssistantPart::Reasoning(llmwire::ReasoningPart {
                    id: Some("rs_1".into()),
                    content: vec![llmwire::ReasoningContent::Encrypted { data: "ENC".into() }],
                    provider_metadata: llmwire::ProviderMetadata::default(),
                }),
                llmwire::AssistantPart::Text {
                    text: "hello".into(),
                    provider_metadata: llmwire::ProviderMetadata::default(),
                },
            ],
            provider_metadata: llmwire::ProviderMetadata::with(
                "llmwire",
                json!({"profile": "openai-responses"}),
            ),
        })
        .message(Message::user("next"))
        .build();

    let result = provider
        .language_model("m")
        .generate(request)
        .await
        .expect("generate succeeds");

    let body = mock.request_json(0);
    let assistant = &body["messages"][1];
    assert_eq!(assistant["content"], "hello");
    assert!(
        assistant.get("reasoning_details").is_none(),
        "foreign encrypted reasoning must not be rebuilt: {assistant}"
    );
    assert!(!body.to_string().contains("ENC"));
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.message.contains("openai-responses")),
        "{:?}",
        result.warnings
    );
}
