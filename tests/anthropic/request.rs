use super::*;

#[tokio::test]
async fn request_golden_with_thinking_replay_and_compaction() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let provider = anthropic(&mock);

    let request = Request::builder()
        .system("Be safe.")
        .message(Message::user("hi"))
        .message(Message::Assistant {
            content: vec![
                AssistantPart::Reasoning(ReasoningPart {
                    id: None,
                    content: vec![ReasoningContent::Text {
                        text: "let me think".into(),
                        signature: Some("SIG1".into()),
                    }],
                    provider_metadata: ProviderMetadata::default(),
                }),
                AssistantPart::Reasoning(ReasoningPart {
                    id: None,
                    content: vec![ReasoningContent::Redacted {
                        data: "RED1".into(),
                    }],
                    provider_metadata: ProviderMetadata::default(),
                }),
                AssistantPart::Text {
                    text: "calling tool".into(),
                    provider_metadata: ProviderMetadata::default(),
                },
            ],
            provider_metadata: ProviderMetadata::default(),
        })
        .message(Message::user("continue"))
        .tools(tool_request("x").tools)
        .tool_choice(ToolChoice::Required)
        .parallel_tool_calls(false)
        .structured_output(StructuredOutput::new(
            "out",
            json!({"type": "object", "additionalProperties": false}),
        ))
        .compaction(Compaction {
            trigger_input_tokens: Some(150_000),
            pause_after_compaction: Some(true),
            instructions: Some("keep tool results".into()),
        })
        .max_output_tokens(32_000)
        .build();

    provider
        .language_model("claude-opus-5")
        .generate(request)
        .await
        .expect("generate succeeds");

    let http = &mock.requests()[0];
    assert_eq!(http.url.as_str(), "https://api.anthropic.com/v1/messages");
    assert_eq!(header(http, "x-api-key"), Some("sk-ant-test"));
    assert_eq!(header(http, "anthropic-version"), Some("2023-06-01"));
    assert_eq!(header(http, "anthropic-beta"), Some("compact-2026-01-12"));

    let body = mock.request_json(0);
    assert_eq!(body["max_tokens"], 32000);
    assert_eq!(body["system"], "Be safe.");
    assert_eq!(body["tool_choice"]["type"], "any");
    assert_eq!(
        body["tool_choice"]["disable_parallel_tool_use"],
        json!(true)
    );
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");
    assert_eq!(
        body["context_management"]["edits"][0]["type"],
        "compact_20260112"
    );
    assert_eq!(
        body["context_management"]["edits"][0]["trigger"],
        json!({"type": "input_tokens", "value": 150000})
    );
    assert_eq!(
        body["context_management"]["edits"][0]["pause_after_compaction"],
        json!(true)
    );

    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "user");
    let assistant_blocks = messages[1]["content"].as_array().unwrap();
    assert_eq!(assistant_blocks[0]["type"], "thinking");
    assert_eq!(assistant_blocks[0]["thinking"], "let me think");
    assert_eq!(assistant_blocks[0]["signature"], "SIG1");
    assert_eq!(assistant_blocks[1]["type"], "redacted_thinking");
    assert_eq!(assistant_blocks[1]["data"], "RED1");
    assert_eq!(assistant_blocks[2]["type"], "text");

    assert_eq!(body["tools"][0]["name"], "get_weather");
    assert!(body["tools"][0].get("input_schema").is_some());
}

#[tokio::test]
async fn swapping_the_history_keeps_the_system_prompt() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let provider = anthropic(&mock);

    let base = Request::builder().system("Be safe.").build();
    let mut request = base.clone();
    request.messages = vec![Message::user("hi")];

    provider
        .language_model("claude-opus-5")
        .generate(request)
        .await
        .expect("generate succeeds");

    let body = mock.request_json(0);
    assert_eq!(body["system"], "Be safe.");
    assert_eq!(body["messages"][0]["role"], "user");
}

#[tokio::test]
async fn unknown_model_gets_fallback_max_tokens() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let provider = anthropic(&mock);
    provider
        .language_model("claude-someday-9")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");
    assert_eq!(mock.request_json(0)["max_tokens"], 8192);
}

#[tokio::test]
async fn user_beta_headers_merge_with_protocol_betas() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let provider = provider_with(
        &mock,
        caido_ai::ProviderConfig::anthropic(caido_ai::Credentials::api_key("k")).with_header(
            HeaderName::from_static("anthropic-beta"),
            HeaderValue::from_static("context-management-2025-06-27"),
        ),
    );
    let request = Request::builder()
        .message(Message::user("hi"))
        .compaction(Compaction::enabled())
        .build();
    provider
        .language_model("claude-opus-5")
        .generate(request)
        .await
        .expect("generate succeeds");
    let http = &mock.requests()[0];
    let beta = header(http, "anthropic-beta").expect("beta header present");
    assert!(
        beta.contains("compact-2026-01-12"),
        "protocol beta kept: {beta}"
    );
    assert!(
        beta.contains("context-management-2025-06-27"),
        "user beta merged: {beta}"
    );
}

#[tokio::test]
async fn consecutive_tool_results_merge_into_one_user_message() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let provider = anthropic(&mock);
    let call_a = caido_ai::ToolCall {
        call_id: "toolu_a".into(),
        item_id: None,
        provider_call_id: None,
        name: "get_weather".into(),
        arguments: "{}".into(),
        provider_metadata: ProviderMetadata::default(),
    };
    let call_b = caido_ai::ToolCall {
        call_id: "toolu_b".into(),
        ..call_a.clone()
    };
    let request = Request::builder()
        .message(Message::user("both"))
        .message(Message::Assistant {
            content: vec![
                AssistantPart::ToolCall(call_a.clone()),
                AssistantPart::ToolCall(call_b.clone()),
            ],
            provider_metadata: ProviderMetadata::default(),
        })
        .message(Message::tool_result(ToolResultPart::for_call(
            &call_a, "21C",
        )))
        .message(Message::tool_result(ToolResultPart::for_call(
            &call_b, "17C",
        )))
        .build();
    provider
        .language_model("claude-opus-5")
        .generate(request)
        .await
        .expect("generate succeeds");
    let messages = mock.request_json(0)["messages"].as_array().unwrap().clone();
    assert_eq!(messages.len(), 3, "{messages:#?}");
    assert_eq!(messages[2]["content"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn malformed_tool_arguments_are_rejected() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let provider = anthropic(&mock);
    let call = caido_ai::ToolCall {
        call_id: "toolu_x".into(),
        item_id: None,
        provider_call_id: None,
        name: "t".into(),
        arguments: "{\"a\":".into(),
        provider_metadata: ProviderMetadata::default(),
    };
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(Message::Assistant {
            content: vec![AssistantPart::ToolCall(call)],
            provider_metadata: ProviderMetadata::default(),
        })
        .build();
    let error = provider
        .language_model("claude-opus-5")
        .generate(request)
        .await
        .expect_err("malformed history must not be changed into an executable call");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert!(mock.requests().is_empty());
}

#[tokio::test]
async fn empty_text_parts_are_not_sent() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(Message::Assistant {
            content: vec![
                caido_ai::AssistantPart::Text {
                    text: String::new(),
                    provider_metadata: ProviderMetadata::default(),
                },
                caido_ai::AssistantPart::Text {
                    text: "kept".into(),
                    provider_metadata: ProviderMetadata::default(),
                },
            ],
            provider_metadata: ProviderMetadata::default(),
        })
        .message(Message::User { content: vec![] })
        .message(Message::user("again"))
        .build();
    anthropic(&mock)
        .language_model("claude-opus-5")
        .generate(request)
        .await
        .unwrap();

    let messages = mock.request_json(0)["messages"].clone();
    assert_eq!(
        messages,
        json!([
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "kept"}]},
            {"role": "user", "content": [{"type": "text", "text": "again"}]},
        ])
    );
}

#[tokio::test]
async fn reasoning_budgets_without_an_output_cap_raise_max_tokens() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let request = Request::builder()
        .message(Message::user("hi"))
        .reasoning(ReasoningConfig::budget(16_000.try_into().unwrap()))
        .build();
    anthropic(&mock)
        .language_model("claude-opus-5")
        .generate(request)
        .await
        .expect("the budget must not be rejected against the SDK fallback cap");

    let body = mock.request_json(0);
    assert_eq!(body["thinking"]["budget_tokens"], 16_000);
    assert_eq!(body["max_tokens"], 8192 + 16_000);
}
