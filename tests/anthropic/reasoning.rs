use super::*;

#[tokio::test]
async fn effort_uses_adaptive_thinking_without_model_inference() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let provider = anthropic(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .reasoning(cogfer::ReasoningConfig::Effort {
            effort: cogfer::ReasoningEffort::High,
            output: Some(cogfer::ReasoningOutput::Include),
        })
        .build();

    provider
        .language_model("caller-selected-model")
        .generate(request)
        .await
        .expect("generate succeeds");

    let body = mock.request_json(0);
    assert_eq!(
        body["thinking"],
        json!({"type": "adaptive", "display": "summarized"})
    );
    assert_eq!(body["output_config"]["effort"], "high");
}

#[tokio::test]
async fn budget_uses_exact_manual_thinking_configuration() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let provider = anthropic(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .reasoning(cogfer::ReasoningConfig::Budget {
            tokens: std::num::NonZeroU32::new(2048).unwrap(),
            output: Some(cogfer::ReasoningOutput::Omit),
        })
        .max_output_tokens(4096)
        .build();

    provider
        .language_model("caller-selected-model")
        .generate(request)
        .await
        .expect("generate succeeds");

    assert_eq!(
        mock.request_json(0)["thinking"],
        json!({"type": "enabled", "budget_tokens": 2048, "display": "omitted"})
    );
}

#[tokio::test]
async fn disabled_reasoning_is_sent_explicitly() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let provider = anthropic(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .reasoning(cogfer::ReasoningConfig::Disabled)
        .build();

    provider
        .language_model("caller-selected-model")
        .generate(request)
        .await
        .expect("generate succeeds");

    assert_eq!(
        mock.request_json(0)["thinking"],
        json!({"type": "disabled"})
    );
}

#[tokio::test]
async fn manual_budget_below_protocol_minimum_is_rejected() {
    let mock = MockTransport::shared();
    let provider = anthropic(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .reasoning(cogfer::ReasoningConfig::budget(
            std::num::NonZeroU32::new(1023).unwrap(),
        ))
        .max_output_tokens(4096)
        .build();

    let error = provider
        .language_model("caller-selected-model")
        .generate(request)
        .await
        .expect_err("invalid budget is rejected");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert!(mock.requests().is_empty());
}

#[tokio::test]
async fn manual_budget_is_not_clamped_to_fit_output_limit() {
    let mock = MockTransport::shared();
    let provider = anthropic(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .reasoning(cogfer::ReasoningConfig::budget(
            std::num::NonZeroU32::new(4096).unwrap(),
        ))
        .max_output_tokens(4096)
        .build();

    let error = provider
        .language_model("caller-selected-model")
        .generate(request)
        .await
        .expect_err("invalid budget is rejected");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert!(mock.requests().is_empty());
}

#[tokio::test]
async fn manual_budget_does_not_rewrite_forced_tool_choice() {
    let mock = MockTransport::shared();
    let provider = anthropic(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .tools(tool_request("x").tools)
        .tool_choice(ToolChoice::Required)
        .reasoning(cogfer::ReasoningConfig::budget(
            std::num::NonZeroU32::new(2048).unwrap(),
        ))
        .max_output_tokens(4096)
        .build();

    let error = provider
        .language_model("caller-selected-model")
        .generate(request)
        .await
        .expect_err("incompatible settings are rejected");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert!(mock.requests().is_empty());
}

#[tokio::test]
async fn foreign_unsigned_reasoning_is_not_replayed_as_thinking() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let provider = anthropic(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(Message::Assistant {
            content: vec![
                AssistantPart::Reasoning(ReasoningPart {
                    id: None,
                    content: vec![ReasoningContent::Text {
                        text: "mine".into(),
                        signature: Some("SIG".into()),
                    }],
                    provider_metadata: ProviderMetadata::default(),
                }),
                AssistantPart::Reasoning(ReasoningPart {
                    id: None,
                    content: vec![ReasoningContent::Text {
                        text: "from another provider".into(),
                        signature: None,
                    }],
                    provider_metadata: ProviderMetadata::default(),
                }),
                AssistantPart::Text {
                    text: "answer".into(),
                    provider_metadata: ProviderMetadata::default(),
                },
            ],
            provider_metadata: ProviderMetadata::default(),
        })
        .message(Message::user("again"))
        .build();

    provider
        .language_model("caller-selected-model")
        .generate(request)
        .await
        .expect("generate succeeds");

    let body = mock.request_json(0);
    let blocks = body["messages"][1]["content"].as_array().unwrap();
    let thinking: Vec<_> = blocks
        .iter()
        .filter(|block| block["type"] == "thinking")
        .collect();
    assert_eq!(thinking.len(), 1);
    assert_eq!(thinking[0]["signature"], "SIG");
}

#[tokio::test]
async fn effort_falls_back_to_budget_when_the_model_has_no_efforts() {
    // A model narrowed to budget-only reasoning (e.g. claude-haiku-4-5, which
    // has no adaptive thinking) maps a requested effort onto a token budget
    // instead of sending `thinking: {"type": "adaptive"}`.
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let profile = cogfer::ApiProfile::AnthropicMessages;
    let capabilities = cogfer::ModelCapabilities {
        reasoning: cogfer::ReasoningSupport {
            efforts: Vec::new(),
            ..cogfer::ModelCapabilities::for_profile(profile).reasoning
        },
        ..cogfer::ModelCapabilities::for_profile(profile)
    };
    let result = anthropic(&mock)
        .language_model("claude-haiku-4-5")
        .with_capabilities(&capabilities)
        .generate(
            Request::builder()
                .message(Message::user("hi"))
                .reasoning(cogfer::ReasoningConfig::effort(
                    cogfer::ReasoningEffort::Medium,
                ))
                .build(),
        )
        .await
        .expect("generate succeeds");

    let body = mock.request_json(0);
    assert_eq!(body["thinking"]["type"], "enabled", "{body}");
    assert_eq!(body["thinking"]["budget_tokens"], 16_384, "{body}");
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.subject.as_deref() == Some("reasoning.effort")),
        "{:?}",
        result.warnings
    );
}

#[tokio::test]
async fn derived_budget_is_fitted_under_the_output_cap() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let profile = cogfer::ApiProfile::AnthropicMessages;
    let capabilities = cogfer::ModelCapabilities {
        reasoning: cogfer::ReasoningSupport {
            efforts: Vec::new(),
            ..cogfer::ModelCapabilities::for_profile(profile).reasoning
        },
        ..cogfer::ModelCapabilities::for_profile(profile)
    };
    let result = anthropic(&mock)
        .language_model("claude-haiku-4-5")
        .with_capabilities(&capabilities)
        .generate(
            Request::builder()
                .message(Message::user("hi"))
                .reasoning(cogfer::ReasoningConfig::effort(
                    cogfer::ReasoningEffort::Medium,
                ))
                .max_output_tokens(2000)
                .build(),
        )
        .await
        .expect("generate succeeds");

    let body = mock.request_json(0);
    assert_eq!(body["max_tokens"], 2000, "{body}");
    assert_eq!(body["thinking"]["budget_tokens"], 1999, "{body}");
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.message.contains("reduced to 1999")),
        "{:?}",
        result.warnings
    );
}

#[tokio::test]
async fn derived_budget_is_dropped_when_the_output_cap_cannot_fit_thinking() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let profile = cogfer::ApiProfile::AnthropicMessages;
    let capabilities = cogfer::ModelCapabilities {
        reasoning: cogfer::ReasoningSupport {
            efforts: Vec::new(),
            ..cogfer::ModelCapabilities::for_profile(profile).reasoning
        },
        ..cogfer::ModelCapabilities::for_profile(profile)
    };
    let result = anthropic(&mock)
        .language_model("claude-haiku-4-5")
        .with_capabilities(&capabilities)
        .generate(
            Request::builder()
                .message(Message::user("hi"))
                .reasoning(cogfer::ReasoningConfig::effort(
                    cogfer::ReasoningEffort::Low,
                ))
                .max_output_tokens(512)
                .build(),
        )
        .await
        .expect("generate succeeds");

    let body = mock.request_json(0);
    assert!(body.get("thinking").is_none(), "{body}");
    assert!(
        result.warnings.iter().any(|warning| warning
            .message
            .contains("cannot fit the 1024-token minimum")),
        "{:?}",
        result.warnings
    );
}

#[tokio::test]
async fn derived_budget_fit_handles_the_minimum_boundaries() {
    let profile = cogfer::ApiProfile::AnthropicMessages;
    let capabilities = cogfer::ModelCapabilities {
        reasoning: cogfer::ReasoningSupport {
            efforts: Vec::new(),
            ..cogfer::ModelCapabilities::for_profile(profile).reasoning
        },
        ..cogfer::ModelCapabilities::for_profile(profile)
    };
    let request_with_max = |max| {
        Request::builder()
            .message(Message::user("hi"))
            .reasoning(cogfer::ReasoningConfig::effort(
                cogfer::ReasoningEffort::Medium,
            ))
            .max_output_tokens(max)
            .build()
    };

    // 1025 is the smallest cap that still fits the 1024-token minimum.
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    anthropic(&mock)
        .language_model("claude-haiku-4-5")
        .with_capabilities(&capabilities)
        .generate(request_with_max(1025))
        .await
        .expect("generate succeeds");
    let body = mock.request_json(0);
    assert_eq!(body["thinking"]["budget_tokens"], 1024, "{body}");

    // 1024 cannot fit a budget strictly below itself; thinking is dropped.
    mock.push_json(200, &minimal_message());
    anthropic(&mock)
        .language_model("claude-haiku-4-5")
        .with_capabilities(&capabilities)
        .generate(request_with_max(1024))
        .await
        .expect("generate succeeds");
    let body = mock.request_json(1);
    assert!(body.get("thinking").is_none(), "{body}");
}

#[tokio::test]
async fn adaptive_thinking_drops_the_samplers_with_warnings() {
    // The library omits sampling settings in both thinking modes.
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let result = anthropic(&mock)
        .language_model("claude-sonnet-4-6")
        .generate(
            Request::builder()
                .message(Message::user("hi"))
                .reasoning(cogfer::ReasoningConfig::effort(
                    cogfer::ReasoningEffort::Low,
                ))
                .temperature(0.5)
                .top_p(0.9)
                .top_k(40)
                .build(),
        )
        .await
        .expect("generate succeeds");

    let body = mock.request_json(0);
    assert_eq!(body["thinking"]["type"], "adaptive", "{body}");
    assert!(body.get("temperature").is_none(), "{body}");
    assert!(body.get("top_p").is_none(), "{body}");
    assert!(body.get("top_k").is_none(), "{body}");
    let subjects: Vec<_> = result
        .warnings
        .iter()
        .filter_map(|warning| warning.subject.as_deref())
        .collect();
    assert_eq!(
        subjects,
        ["temperature", "top_p", "top_k"],
        "{:?}",
        result.warnings
    );
}

#[tokio::test]
async fn adaptive_thinking_keeps_forced_tool_choice() {
    // Unlike manual budgets, adaptive thinking accepts forced tool choice
    // (verified live against the API); it must not be rejected locally.
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    anthropic(&mock)
        .language_model("claude-sonnet-4-6")
        .generate(
            Request::builder()
                .message(Message::user("hi"))
                .tools(tool_request("x").tools)
                .tool_choice(ToolChoice::Required)
                .reasoning(cogfer::ReasoningConfig::effort(
                    cogfer::ReasoningEffort::Low,
                ))
                .build(),
        )
        .await
        .expect("generate succeeds");

    let body = mock.request_json(0);
    assert_eq!(body["tool_choice"]["type"], "any", "{body}");
    assert_eq!(body["thinking"]["type"], "adaptive", "{body}");
}

#[tokio::test]
async fn disabled_thinking_keeps_the_samplers() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    anthropic(&mock)
        .language_model("claude-sonnet-4-6")
        .generate(
            Request::builder()
                .message(Message::user("hi"))
                .reasoning(cogfer::ReasoningConfig::Disabled)
                .temperature(0.5)
                .build(),
        )
        .await
        .expect("generate succeeds");
    assert_eq!(mock.request_json(0)["temperature"], 0.5);
}

#[tokio::test]
async fn dropped_derived_budget_re_enables_the_samplers() {
    // effort + tiny cap: the derived budget is dropped, so no thinking goes
    // on the wire and the samplers must come back.
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let profile = cogfer::ApiProfile::AnthropicMessages;
    let capabilities = cogfer::ModelCapabilities {
        reasoning: cogfer::ReasoningSupport {
            efforts: Vec::new(),
            ..cogfer::ModelCapabilities::for_profile(profile).reasoning
        },
        ..cogfer::ModelCapabilities::for_profile(profile)
    };
    anthropic(&mock)
        .language_model("claude-haiku-4-5")
        .with_capabilities(&capabilities)
        .generate(
            Request::builder()
                .message(Message::user("hi"))
                .reasoning(cogfer::ReasoningConfig::effort(
                    cogfer::ReasoningEffort::Low,
                ))
                .max_output_tokens(512)
                .temperature(0.5)
                .build(),
        )
        .await
        .expect("generate succeeds");

    let body = mock.request_json(0);
    assert!(body.get("thinking").is_none(), "{body}");
    assert_eq!(body["temperature"], 0.5, "{body}");
}
