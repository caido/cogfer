use super::*;

#[tokio::test]
async fn effort_uses_adaptive_thinking_without_model_inference() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let provider = anthropic(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .reasoning(llmwire::ReasoningConfig::Effort {
            effort: llmwire::ReasoningEffort::High,
            output: Some(llmwire::ReasoningOutput::Include),
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
        .reasoning(llmwire::ReasoningConfig::Budget {
            tokens: std::num::NonZeroU32::new(2048).unwrap(),
            output: Some(llmwire::ReasoningOutput::Omit),
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
        .reasoning(llmwire::ReasoningConfig::Disabled)
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
        .reasoning(llmwire::ReasoningConfig::budget(
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
        .reasoning(llmwire::ReasoningConfig::budget(
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
        .reasoning(llmwire::ReasoningConfig::budget(
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
