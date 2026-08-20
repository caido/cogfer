use super::*;

#[tokio::test]
async fn request_golden_with_signature_replay() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_response());
    let provider = gemini(&mock);

    let call = ToolCall {
        call_id: "fc_1".into(),
        item_id: Some("fc_1".into()),
        provider_call_id: Some("fc_1".into()),
        name: "get_weather".into(),
        arguments: "{\"location\":\"Paris\"}".into(),
        provider_metadata: ProviderMetadata::with(
            "gemini",
            json!({"thought_signature": "SIGBYTES"}),
        ),
    };
    let request = Request::builder()
        .system("Be brief.")
        .message(Message::user("Weather in Paris and London?"))
        .message(Message::Assistant {
            content: vec![AssistantPart::ToolCall(call.clone())],
            provider_metadata: ProviderMetadata::default(),
        })
        .message(Message::tool_result(ToolResultPart::json_for_call(
            &call,
            json!({"result": {"temp_c": 21}}),
        )))
        .tools(tool_request("x").tools)
        .structured_output(StructuredOutput::new(
            "out",
            json!({"type": "object", "properties": {"a": {"type": "string"}}}),
        ))
        .reasoning(ReasoningConfig::Budget {
            tokens: std::num::NonZeroU32::new(1024).unwrap(),
            output: Some(ReasoningOutput::Include),
        })
        .max_output_tokens(300)
        .temperature(0.5)
        .build();

    provider
        .language_model("gemini-2.5-flash")
        .generate(request)
        .await
        .expect("generate succeeds");

    let http = &mock.requests()[0];
    assert_eq!(
        http.url.as_str(),
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    );
    assert!(
        http.headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("x-goog-api-key") && value == "g-test"
        })
    );

    let body = mock.request_json(0);
    assert_eq!(body["systemInstruction"]["parts"][0]["text"], "Be brief.");
    let contents = body["contents"].as_array().unwrap();
    assert_eq!(contents[0]["role"], "user");
    assert_eq!(contents[1]["role"], "model");
    assert_eq!(
        contents[1]["parts"][0]["functionCall"]["name"],
        "get_weather"
    );
    assert_eq!(contents[1]["parts"][0]["functionCall"]["id"], "fc_1");
    assert_eq!(contents[1]["parts"][0]["thoughtSignature"], "SIGBYTES");
    assert_eq!(contents[2]["role"], "user");
    assert_eq!(
        contents[2]["parts"][0]["functionResponse"]["name"],
        "get_weather"
    );
    assert_eq!(contents[2]["parts"][0]["functionResponse"]["id"], "fc_1");
    assert_eq!(
        contents[2]["parts"][0]["functionResponse"]["response"],
        json!({"result": {"temp_c": 21}})
    );

    assert!(
        body["tools"][0]["functionDeclarations"][0]
            .get("parametersJsonSchema")
            .is_some()
    );
    assert_eq!(
        body["generationConfig"]["responseMimeType"],
        "application/json"
    );
    assert!(body["generationConfig"].get("responseJsonSchema").is_some());
    assert_eq!(
        body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
        1024
    );
    assert_eq!(
        body["generationConfig"]["thinkingConfig"]["includeThoughts"],
        json!(true)
    );
    assert_eq!(body["generationConfig"]["maxOutputTokens"], 300);
}

#[tokio::test]
async fn effort_is_sent_as_a_lowercase_thinking_level() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_response());
    let provider = gemini(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .reasoning(ReasoningConfig::effort(ReasoningEffort::High))
        .build();
    provider
        .language_model("gemini-3-flash")
        .generate(request)
        .await
        .expect("generate succeeds");
    assert_eq!(
        mock.request_json(0)["generationConfig"]["thinkingConfig"]["thinkingLevel"],
        "high"
    );
}

#[tokio::test]
async fn disabled_reasoning_is_sent_as_zero_budget() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_response());
    let provider = gemini(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .reasoning(ReasoningConfig::Disabled)
        .build();

    provider
        .language_model("caller-selected-model")
        .generate(request)
        .await
        .expect("generate succeeds");

    assert_eq!(
        mock.request_json(0)["generationConfig"]["thinkingConfig"],
        json!({"thinkingBudget": 0})
    );
}

#[tokio::test]
async fn consecutive_tool_results_merge_into_one_turn() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_response());
    let provider = gemini(&mock);

    let call_a = ToolCall {
        call_id: "fc_a".into(),
        item_id: Some("fc_a".into()),
        provider_call_id: Some("fc_a".into()),
        name: "get_weather".into(),
        arguments: "{\"location\":\"Paris\"}".into(),
        provider_metadata: ProviderMetadata::default(),
    };
    let call_b = ToolCall {
        call_id: "fc_b".into(),
        item_id: Some("fc_b".into()),
        provider_call_id: Some("fc_b".into()),
        name: "get_weather".into(),
        arguments: "{\"location\":\"London\"}".into(),
        provider_metadata: ProviderMetadata::default(),
    };
    let request = Request::builder()
        .message(Message::user("both?"))
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
        .language_model("gemini-2.5-flash")
        .generate(request)
        .await
        .expect("generate succeeds");

    let contents = mock.request_json(0)["contents"].as_array().unwrap().clone();
    assert_eq!(contents.len(), 3);
    let responses = contents[2]["parts"].as_array().unwrap();
    assert_eq!(responses.len(), 2);
    assert!(
        responses
            .iter()
            .all(|part| part.get("functionResponse").is_some())
    );
}

#[tokio::test]
async fn stream_url_uses_alt_sse() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"responseId":"r6"}"#,
    ]);
    let provider = gemini(&mock);
    provider
        .language_model("gemini-2.5-flash")
        .stream(text_request("hi"))
        .await
        .unwrap()
        .collect_result()
        .await
        .expect("collect");
    assert_eq!(
        mock.requests()[0].url.as_str(),
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    );
}

#[tokio::test]
async fn non_object_tool_arguments_are_rejected() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_response());
    let provider = gemini(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(Message::Assistant {
            content: vec![AssistantPart::ToolCall(ToolCall {
                call_id: "fc_9".into(),
                item_id: None,
                provider_call_id: None,
                name: "lookup".into(),
                arguments: "null".into(),
                provider_metadata: ProviderMetadata::default(),
            })],
            provider_metadata: ProviderMetadata::default(),
        })
        .build();
    let error = provider
        .language_model("gemini-2.5-flash")
        .generate(request)
        .await
        .expect_err("non-object history must not be changed into an executable call");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert!(mock.requests().is_empty());
}

#[tokio::test]
async fn strict_tool_schema_emits_an_unsupported_setting_warning() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_response());
    let provider = gemini(&mock);
    let mut tool = tool_request("x").tools.into_iter().next().unwrap();
    tool.strict = Some(true);
    let request = Request::builder()
        .message(Message::user("hi"))
        .tool(tool)
        .build();

    let result = provider
        .language_model("gemini-2.5-flash")
        .generate(request)
        .await
        .expect("generate succeeds");

    let warning = result
        .warnings
        .iter()
        .find(|warning| warning.subject.as_deref() == Some("tools.strict"))
        .expect("strict omission is reported");
    assert_eq!(warning.kind, caido_ai::WarningKind::UnsupportedSetting);
    assert!(
        mock.request_json(0)["tools"][0]["functionDeclarations"][0]
            .get("strict")
            .is_none()
    );
}
