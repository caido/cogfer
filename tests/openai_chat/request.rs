use super::*;

#[tokio::test]
async fn request_golden() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completion());
    let provider = openai_chat(&mock);

    let call = ToolCall {
        call_id: "call_a".into(),
        item_id: None,
        name: "get_weather".into(),
        arguments: "{\"location\":\"Paris\"}".into(),
        provider_metadata: ProviderMetadata::default(),
    };
    let request = Request::builder()
        .system("Be helpful.")
        .message(Message::user("Weather?"))
        .message(Message::Assistant {
            content: vec![llmwire::AssistantPart::ToolCall(call.clone())],
            provider_metadata: ProviderMetadata::default(),
        })
        .message(Message::tool_result(ToolResultPart::for_call(&call, "21C")))
        .tools(tool_request("x").tools)
        .reasoning(ReasoningConfig::effort(ReasoningEffort::High))
        .structured_output(StructuredOutput::new(
            "out",
            json!({"type": "object", "additionalProperties": false}),
        ))
        .max_output_tokens(200)
        .stop_sequence("END")
        .build();

    provider
        .language_model("caller-selected-model")
        .generate(request)
        .await
        .expect("generate succeeds");

    let http = &mock.requests()[0];
    assert_eq!(
        http.url.as_str(),
        "https://api.openai.com/v1/chat/completions"
    );
    let body = mock.request_json(0);
    assert_eq!(body["max_completion_tokens"], 200);
    assert!(body.get("max_tokens").is_none());
    assert_eq!(body["reasoning_effort"], "high");
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(body["stop"], json!(["END"]));

    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[2]["role"], "assistant");
    assert_eq!(messages[2]["tool_calls"][0]["id"], "call_a");
    assert_eq!(messages[2]["tool_calls"][0]["type"], "function");
    assert_eq!(messages[2]["content"], serde_json::Value::Null);
    assert_eq!(messages[3]["role"], "tool");
    assert_eq!(messages[3]["tool_call_id"], "call_a");
    assert_eq!(messages[3]["content"], "21C");

    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
}

#[tokio::test]
async fn disabled_reasoning_is_sent_explicitly() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completion());
    let provider = openai_chat(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .reasoning(ReasoningConfig::Disabled)
        .build();
    provider
        .language_model("model")
        .generate(request)
        .await
        .expect("generate succeeds");
    let body = mock.request_json(0);
    assert_eq!(body["reasoning_effort"], json!("none"));
}

#[tokio::test]
async fn opaque_compaction_part_is_rejected() {
    let mock = MockTransport::shared();
    let provider = openai_chat(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(assistant_with(vec![llmwire::AssistantPart::Compaction(
            llmwire::CompactionPart {
                id: Some("cmp_1".into()),
                content: None,
                encrypted_content: Some("OPAQUE".into()),
            },
        )]))
        .message(Message::user("next"))
        .build();
    let error = provider
        .language_model("gpt-4o")
        .generate(request)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
    assert!(mock.requests().is_empty(), "no request should be sent");
}

#[tokio::test]
async fn summary_compaction_part_is_flattened_with_warning() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completion());
    let provider = openai_chat(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(assistant_with(vec![llmwire::AssistantPart::Compaction(
            llmwire::CompactionPart {
                id: None,
                content: Some("Earlier we discussed pricing.".into()),
                encrypted_content: None,
            },
        )]))
        .message(Message::user("next"))
        .build();
    let result = provider
        .language_model("gpt-4o")
        .generate(request)
        .await
        .expect("generate succeeds");
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.message.contains("compaction")),
        "{:?}",
        result.warnings
    );
    let body = mock.request_json(0);
    assert_eq!(
        body["messages"][1]["content"],
        json!("Earlier we discussed pricing.")
    );
}

#[tokio::test]
async fn reasoning_only_assistant_turn_is_skipped() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completion());
    let provider = openai_chat(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(assistant_with(vec![llmwire::AssistantPart::Reasoning(
            llmwire::ReasoningPart {
                id: Some("rs_1".into()),
                content: vec![llmwire::ReasoningContent::Encrypted {
                    data: "BLOB".into(),
                }],
                provider_metadata: ProviderMetadata::default(),
            },
        )]))
        .message(Message::user("next"))
        .build();
    provider
        .language_model("gpt-4o")
        .generate(request)
        .await
        .expect("generate succeeds");
    let body = mock.request_json(0);
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2, "{messages:?}");
    assert!(messages.iter().all(|m| m["role"] != "assistant"));
}

#[tokio::test]
async fn multiple_user_text_parts_stay_separate_content_parts() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completion());
    let request = Request::builder()
        .message(Message::User {
            content: vec![
                llmwire::UserPart::Text {
                    text: "Summarize:".into(),
                },
                llmwire::UserPart::Text {
                    text: "<body>".into(),
                },
            ],
        })
        .build();
    openai_chat(&mock)
        .language_model("gpt-5.6")
        .generate(request)
        .await
        .unwrap();

    let body = mock.request_json(0);
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {"type": "text", "text": "Summarize:"},
            {"type": "text", "text": "<body>"}
        ])
    );
    // A single part keeps the plain-string form every server accepts.
    mock.push_json(200, &minimal_completion());
    openai_chat(&mock)
        .language_model("gpt-5.6")
        .generate(text_request("hi"))
        .await
        .unwrap();
    assert_eq!(mock.request_json(1)["messages"][0]["content"], "hi");
}

#[tokio::test]
async fn compatible_servers_get_the_portable_wire_spellings() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completion());
    mock.push_sse(&["[DONE]"]);
    let provider = provider_with(
        &mock,
        llmwire::ProviderConfig::openai_chat(llmwire::Credentials::none())
            .with_base_url("http://localhost:11434/v1".parse().unwrap()),
    );
    let request = Request::builder()
        .message(Message::user("hi"))
        .max_output_tokens(64)
        .build();
    provider
        .language_model("qwen3.5")
        .generate(request.clone())
        .await
        .unwrap();
    drain(
        provider
            .language_model("qwen3.5")
            .stream(request)
            .await
            .unwrap(),
    )
    .await;

    let generate = mock.request_json(0);
    assert_eq!(generate["max_tokens"], 64);
    assert!(generate.get("max_completion_tokens").is_none());
    let stream = mock.request_json(1);
    assert_eq!(stream["stream_options"], json!({"include_usage": true}));

    // OpenAI itself gets the full wire format.
    mock.push_json(200, &minimal_completion());
    openai_chat(&mock)
        .language_model("gpt-5.6")
        .generate(
            Request::builder()
                .message(Message::user("hi"))
                .max_output_tokens(64)
                .build(),
        )
        .await
        .unwrap();
    assert_eq!(mock.request_json(2)["max_completion_tokens"], 64);
}

#[tokio::test]
async fn dialect_downgrade_is_surfaced_when_the_output_cap_changes_spelling() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completion());
    let provider = provider_with(
        &mock,
        llmwire::ProviderConfig::openai_chat(llmwire::Credentials::none())
            .with_base_url("http://localhost:11434/v1".parse().unwrap()),
    );
    let result = provider
        .language_model("qwen3.5")
        .generate(
            Request::builder()
                .message(Message::user("hi"))
                .max_output_tokens(64)
                .build(),
        )
        .await
        .expect("generate succeeds");
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.message.contains("max_completion_tokens")),
        "{:?}",
        result.warnings
    );

    // Without an output cap the downgrade changes nothing worth surfacing.
    mock.push_json(200, &minimal_completion());
    let result = provider
        .language_model("qwen3.5")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
}

#[tokio::test]
async fn compatible_reasoning_survives_tool_replay_without_changing_its_wire_field() {
    for field in ["reasoning_content", "reasoning"] {
        for streaming in [false, true] {
            let mock = MockTransport::shared();
            let message = json!({
                field: "I need the weather.",
                "tool_calls": [{"index": 0, "id": "call_weather", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}}]
            });
            if streaming {
                mock.push_sse(&[
                    &json!({"choices": [{"index": 0, "delta": message,
                        "finish_reason": "tool_calls"}]})
                    .to_string(),
                    "[DONE]",
                ]);
            } else {
                mock.push_json(
                    200,
                    &json!({"choices": [{"message": message,
                    "finish_reason": "tool_calls"}]}),
                );
            }
            let provider = provider_with(
                &mock,
                llmwire::ProviderConfig::openai_chat(llmwire::Credentials::none())
                    .with_base_url("https://api.deepseek.com".parse().unwrap()),
            );
            let model = provider.language_model("deepseek-v4-pro");
            let result = if streaming {
                model
                    .stream(tool_request("Weather?"))
                    .await
                    .unwrap()
                    .collect_result()
                    .await
                    .unwrap()
            } else {
                model.generate(tool_request("Weather?")).await.unwrap()
            };
            let call = result.tool_calls().next().expect("weather call");
            let mut request = tool_request("Weather?");
            request.messages.push(result.to_assistant_message());
            request
                .messages
                .push(Message::tool_result(ToolResultPart::for_call(call, "21C")));
            mock.push_json(200, &minimal_completion());
            model.generate(request.clone()).await.unwrap();
            let body = mock.request_json(1);
            assert_eq!(
                body["messages"][1][field], "I need the weather.",
                "{field}, streaming={streaming}"
            );
            let other = if field == "reasoning" { "reasoning_content" } else { "reasoning" };
            assert!(body["messages"][1].get(other).is_none());

            // OpenAI does not accept the compatible server's extra message field.
            mock.push_json(200, &minimal_completion());
            openai_chat(&mock)
                .language_model("gpt-5.6")
                .generate(request.clone())
                .await
                .unwrap();
            assert!(mock.request_json(2)["messages"][1].get(field).is_none());

            if let Message::Assistant {
                provider_metadata, ..
            } = &mut request.messages[1]
            {
                provider_metadata.insert("llmwire", json!({"profile": "anthropic"}));
            }
            mock.push_json(200, &minimal_completion());
            model.generate(request).await.unwrap();
            assert!(mock.request_json(3)["messages"][1].get(field).is_none());
        }
    }
}
