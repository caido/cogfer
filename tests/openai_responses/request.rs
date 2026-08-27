use super::*;

#[tokio::test]
async fn request_golden() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completed());
    let provider = openai_responses(&mock);
    let model = provider.language_model("gpt-5.6");

    let call = ToolCall {
        call_id: "call_1".into(),
        item_id: Some("fc_1".into()),
        name: "get_weather".into(),
        arguments: "{\"location\":\"Paris\"}".into(),
        provider_metadata: ProviderMetadata::default(),
    };
    let request = Request::builder()
        .system("Be helpful.")
        .message(Message::user("Weather in Paris?"))
        .message(Message::Assistant {
            content: vec![
                AssistantPart::Reasoning(ReasoningPart {
                    id: Some("rs_1".into()),
                    content: vec![
                        ReasoningContent::Summary {
                            text: "thinking".into(),
                        },
                        ReasoningContent::Encrypted { data: "ENC".into() },
                    ],
                    provider_metadata: ProviderMetadata::default(),
                }),
                AssistantPart::ToolCall(call.clone()),
            ],
            provider_metadata: ProviderMetadata::default(),
        })
        .message(Message::tool_result(ToolResultPart::for_call(&call, "21C")))
        .tools(tool_request("x").tools)
        .reasoning(ReasoningConfig::Effort {
            effort: ReasoningEffort::Medium,
            output: Some(ReasoningOutput::Include),
        })
        .structured_output(StructuredOutput::new(
            "weather",
            json!({"type": "object", "properties": {"summary": {"type": "string"}},
                   "required": ["summary"], "additionalProperties": false}),
        ))
        .compaction(Compaction {
            trigger_input_tokens: Some(200_000),
            ..Compaction::enabled()
        })
        .max_output_tokens(500)
        .temperature(0.2)
        .top_k(40)
        .seed(7)
        .build();

    let result = model.generate(request).await.expect("generate succeeds");

    let warned: Vec<_> = result
        .warnings
        .iter()
        .filter_map(|warning| warning.subject.as_deref())
        .collect();
    assert!(warned.contains(&"top_k"));
    assert!(warned.contains(&"seed"));

    let http = &mock.requests()[0];
    assert_eq!(http.url.as_str(), "https://api.openai.com/v1/responses");
    assert_eq!(header(http, "authorization"), Some("Bearer sk-test"));

    let body = mock.request_json(0);
    assert_eq!(body["model"], "gpt-5.6");
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(body["reasoning"]["effort"], "medium");
    assert_eq!(body["reasoning"]["summary"], "auto");
    assert_eq!(body["text"]["format"]["type"], "json_schema");
    assert_eq!(body["text"]["format"]["name"], "weather");
    assert_eq!(body["text"]["format"]["strict"], json!(true));
    assert_eq!(
        body["context_management"],
        json!([{"type": "compaction", "compact_threshold": 200000}])
    );
    assert_eq!(body["max_output_tokens"], 500);
    assert_eq!(body["temperature"], 0.2);
    assert!(body.get("top_k").is_none());
    assert!(body.get("seed").is_none());

    let input = body["input"].as_array().unwrap();
    assert_eq!(input[0]["role"], "system");
    assert_eq!(input[1]["role"], "user");
    assert_eq!(input[2]["type"], "reasoning");
    assert_eq!(input[2]["id"], "rs_1");
    assert_eq!(input[2]["summary"][0]["text"], "thinking");
    assert_eq!(input[2]["encrypted_content"], "ENC");
    assert_eq!(input[3]["type"], "function_call");
    assert_eq!(input[3]["call_id"], "call_1");
    assert_eq!(input[3]["id"], "fc_1");
    assert_eq!(input[4]["type"], "function_call_output");
    assert_eq!(input[4]["call_id"], "call_1");
    assert_eq!(input[4]["output"], "21C");

    let tools = body["tools"].as_array().unwrap();
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["name"], "get_weather");
}

#[tokio::test]
async fn provider_options_escape_hatch_overrides_body() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completed());
    let provider = openai_responses(&mock);
    let model = provider.language_model("gpt-5.6");
    let request = Request::builder()
        .message(Message::user("hi"))
        .provider_option("openai", json!({"store": true, "service_tier": "flex"}))
        .build();
    model.generate(request).await.expect("generate succeeds");
    let body = mock.request_json(0);
    assert_eq!(body["store"], json!(true));
    assert_eq!(body["service_tier"], "flex");
}

#[tokio::test]
async fn omitted_reasoning_output_does_not_request_a_summary() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completed());
    let provider = openai_responses(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .reasoning(ReasoningConfig::Effort {
            effort: ReasoningEffort::Low,
            output: Some(ReasoningOutput::Omit),
        })
        .build();

    provider
        .language_model("model")
        .generate(request)
        .await
        .expect("generate succeeds");

    let reasoning = &mock.request_json(0)["reasoning"];
    assert_eq!(reasoning["effort"], "low");
    assert!(reasoning.get("summary").is_none());
}

#[tokio::test]
async fn disabled_reasoning_is_sent_as_none_effort() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completed());
    let provider = openai_responses(&mock);
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

/// Responses never needs `name` or `item_id` on a result, unlike Gemini.
#[tokio::test]
async fn bare_tool_result_lowers_through_call_id_alone() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completed());
    let provider = openai_responses(&mock);
    let call = ToolCall {
        call_id: "call_1".into(),
        item_id: Some("fc_1".into()),
        name: "lookup".into(),
        arguments: r#"{"query":"value"}"#.into(),
        provider_metadata: ProviderMetadata::default(),
    };
    // A bare result: only the correlation id, as a host that keeps no
    // provider identity would send it.
    let mut result = ToolResultPart::for_call(&call, "found");
    result.item_id = None;
    result.name = None;
    let request = Request::builder()
        .message(Message::Assistant {
            content: vec![AssistantPart::ToolCall(call)],
            provider_metadata: ProviderMetadata::default(),
        })
        .message(Message::tool_result(result))
        .build();

    provider
        .language_model("model")
        .generate(request)
        .await
        .expect("generate succeeds");

    let body = mock.request_json(0);
    assert_eq!(body["input"][1]["type"], "function_call_output");
    assert_eq!(body["input"][1]["call_id"], "call_1");
}

#[tokio::test]
async fn orphan_tool_result_keeps_its_self_contained_call_id() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completed());
    let provider = openai_responses(&mock);
    let call = ToolCall {
        call_id: "call_from_previous_response".into(),
        item_id: None,
        name: "lookup".into(),
        arguments: "{}".into(),
        provider_metadata: ProviderMetadata::default(),
    };
    let mut result = ToolResultPart::for_call(&call, "found");
    result.name = None;
    let request = Request::builder()
        .message(Message::tool_result(result))
        .provider_option("openai", json!({"previous_response_id": "resp_previous"}))
        .build();

    provider
        .language_model("model")
        .generate(request)
        .await
        .expect("generate succeeds");

    let body = mock.request_json(0);
    assert_eq!(body["previous_response_id"], "resp_previous");
    assert_eq!(body["input"][0]["call_id"], "call_from_previous_response");
}

/// A turn full of state only its producing backend can verify: an encrypted
/// reasoning item, a summary-only reasoning item, a phase-tagged text, a
/// server tool payload, and an item-id-bearing tool call.
fn opaque_turn(origin: ProviderMetadata) -> (ToolCall, Message) {
    let call = ToolCall {
        call_id: "call_1".into(),
        item_id: Some("fc_1".into()),
        name: "get_weather".into(),
        arguments: "{}".into(),
        provider_metadata: ProviderMetadata::default(),
    };
    let message = Message::Assistant {
        content: vec![
            AssistantPart::Reasoning(ReasoningPart {
                id: Some("rs_1".into()),
                content: vec![ReasoningContent::Encrypted { data: "ENC".into() }],
                provider_metadata: ProviderMetadata::default(),
            }),
            // Summary-only reasoning is unreplayable on every path and never
            // produces an item or a warning.
            AssistantPart::Reasoning(ReasoningPart {
                id: None,
                content: vec![ReasoningContent::Summary {
                    text: "planning".into(),
                }],
                provider_metadata: ProviderMetadata::default(),
            }),
            AssistantPart::Text {
                text: "checking".into(),
                provider_metadata: ProviderMetadata::with("openai", json!({"phase": "commentary"})),
            },
            AssistantPart::ProviderTool {
                provider_tool: ProviderToolPart {
                    id: Some("ws_1".into()),
                    kind: "web_search_call".into(),
                    namespace: "openai".into(),
                    payload: json!({"type": "web_search_call", "id": "ws_1", "status": "completed"}),
                },
            },
            AssistantPart::ToolCall(call.clone()),
        ],
        provider_metadata: origin,
    };
    (call, message)
}

fn origin(profile: &str) -> ProviderMetadata {
    ProviderMetadata::with("caido-ai", json!({"profile": profile}))
}

#[tokio::test]
async fn foreign_profile_turn_drops_unverifiable_state() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completed());
    let provider = openai_responses(&mock);
    let (call, turn) = opaque_turn(origin("chatgpt"));
    let request = Request::builder()
        .message(Message::user("weather?"))
        .message(turn)
        .message(Message::tool_result(ToolResultPart::for_call(&call, "21C")))
        .build();

    let result = provider
        .language_model("gpt-5.6")
        .generate(request)
        .await
        .expect("generate succeeds");

    let input = mock.request_json(0)["input"].clone();
    let items = input.as_array().unwrap();
    assert!(
        items
            .iter()
            .all(|item| item["type"] != json!("reasoning")
                && item["type"] != json!("web_search_call")),
        "opaque foreign items must not be replayed: {items:?}"
    );
    let function_call = items
        .iter()
        .find(|item| item["type"] == json!("function_call"))
        .expect("tool call still replays");
    assert_eq!(function_call["call_id"], "call_1");
    assert!(
        function_call.get("id").is_none(),
        "foreign item id must be omitted"
    );
    let text = items
        .iter()
        .find(|item| item["role"] == json!("assistant"))
        .expect("text still replays");
    assert_eq!(text["content"][0]["text"], "checking");
    assert!(
        text.get("phase").is_none(),
        "foreign phase marker must be omitted"
    );
    // One warning per foreign origin, however many of its items were dropped.
    assert_eq!(result.warnings.len(), 1, "{:?}", result.warnings);
    assert!(result.warnings[0].message.contains("chatgpt"));
}

#[tokio::test]
async fn native_profile_turn_replays_opaque_state() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completed());
    let provider = openai_responses(&mock);
    let (call, turn) = opaque_turn(origin("openai-responses"));
    let request = Request::builder()
        .message(Message::user("weather?"))
        .message(turn)
        .message(Message::tool_result(ToolResultPart::for_call(&call, "21C")))
        .build();

    let result = provider
        .language_model("gpt-5.6")
        .generate(request)
        .await
        .expect("generate succeeds");

    let input = mock.request_json(0)["input"].clone();
    let items = input.as_array().unwrap();
    let reasoning = items
        .iter()
        .find(|item| item["type"] == json!("reasoning"))
        .expect("native reasoning replays");
    assert_eq!(reasoning["id"], "rs_1");
    assert_eq!(reasoning["encrypted_content"], "ENC");
    let function_call = items
        .iter()
        .find(|item| item["type"] == json!("function_call"))
        .unwrap();
    assert_eq!(function_call["id"], "fc_1");
    let text = items
        .iter()
        .find(|item| item["role"] == json!("assistant"))
        .unwrap();
    assert_eq!(text["phase"], "commentary");
    assert!(
        items
            .iter()
            .any(|item| item["type"] == json!("web_search_call")),
        "native server tool item replays"
    );
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
}

#[tokio::test]
async fn foreign_compaction_cannot_be_replayed() {
    let mock = MockTransport::shared();
    let provider = openai_responses(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(Message::Assistant {
            content: vec![AssistantPart::Compaction(CompactionPart {
                id: Some("cmp_1".into()),
                content: None,
                encrypted_content: Some("OPAQUE".into()),
            })],
            provider_metadata: origin("chatgpt"),
        })
        .message(Message::user("next"))
        .build();

    let error = provider
        .language_model("gpt-5.6")
        .generate(request)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::UnsupportedContent);
    assert!(mock.requests().is_empty());
}

#[tokio::test]
async fn openrouter_reasoning_metadata_marks_the_part_as_foreign() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completed());
    let provider = openai_responses(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(Message::Assistant {
            content: vec![
                AssistantPart::Reasoning(ReasoningPart {
                    id: None,
                    content: vec![ReasoningContent::Encrypted { data: "OR".into() }],
                    provider_metadata: ProviderMetadata::with(
                        "openrouter",
                        json!({"reasoning_details": [{"type": "reasoning.encrypted", "data": "OR"}]}),
                    ),
                }),
                AssistantPart::Text {
                    text: "done".into(),
                    provider_metadata: ProviderMetadata::default(),
                },
            ],
            provider_metadata: ProviderMetadata::default(),
        })
        .message(Message::user("next"))
        .build();

    let result = provider
        .language_model("gpt-5.6")
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
        "openrouter reasoning must not be replayed: {input}"
    );
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.message.contains("openrouter")),
        "{:?}",
        result.warnings
    );
}
