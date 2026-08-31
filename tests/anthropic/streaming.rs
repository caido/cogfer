use super::*;

#[tokio::test]
async fn streaming_thinking_tools_and_cumulative_usage() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_3","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":472,"output_tokens":2}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"User wants weather."}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBCg=="}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Checking."}}"#,
        r#"{"type":"content_block_stop","index":1}"#,
        r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_9","name":"get_weather","input":{}}}"#,
        r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":""}}"#,
        r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}"#,
        r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":" \"Paris\"}"}}"#,
        r#"{"type":"content_block_stop","index":2}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let provider = anthropic(&mock);
    let events = drain(
        provider
            .language_model("claude-opus-5")
            .stream(tool_request("weather"))
            .await
            .unwrap(),
    )
    .await;

    assert_terminal_contract(&events);
    let reasoning = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ReasoningEnd { part, .. } => Some(part.clone()),
            _ => None,
        })
        .expect("reasoning end");
    assert!(matches!(
        &reasoning.content[0],
        ReasoningContent::Text { text, signature: Some(signature) }
            if text == "User wants weather." && signature == "EqQBCg=="
    ));

    let call = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .expect("tool call");
    assert_eq!(call.call_id, "toolu_9");
    assert_eq!(call.arguments, "{\"location\": \"Paris\"}");

    let Some(StreamEvent::Finish { finish, usage }) = events.last() else {
        panic!()
    };
    assert_eq!(finish.reason, FinishReason::ToolCalls);
    assert_eq!(usage.input_tokens, Some(472));
    assert_eq!(usage.output_tokens, Some(89));
}

#[tokio::test]
async fn deferred_tool_call_with_complete_start_input() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_4","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_d","name":"get_weather","input":{"location":"Berlin"}}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let provider = anthropic(&mock);
    let events = drain(
        provider
            .language_model("claude-opus-5")
            .stream(tool_request("weather"))
            .await
            .unwrap(),
    )
    .await;
    assert_terminal_contract(&events);
    let call = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .expect("tool call");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&call.arguments).unwrap(),
        json!({"location": "Berlin"})
    );
}

#[tokio::test]
async fn empty_tool_input_becomes_empty_object() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_5","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_e","name":"list_all","input":{}}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":3}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let provider = anthropic(&mock);
    let events = drain(
        provider
            .language_model("claude-opus-5")
            .stream(tool_request("x"))
            .await
            .unwrap(),
    )
    .await;
    let call = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .expect("tool call");
    assert_eq!(call.arguments, "{}");
}

#[tokio::test]
async fn max_tokens_does_not_promote_a_completed_tool_block() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_partial","content":[]}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_partial","name":"delete","input":{}}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"/tmp\"}"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"max_tokens"}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let events = drain(
        anthropic(&mock)
            .language_model("claude-opus-5")
            .stream(tool_request("x"))
            .await
            .unwrap(),
    )
    .await;

    assert_terminal_contract(&events);
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::ToolCall(_)))
    );
    let Some(StreamEvent::Finish { finish, .. }) = events.last() else {
        panic!("missing finish");
    };
    assert_eq!(finish.reason, FinishReason::Length);
}

#[tokio::test]
async fn invalid_completed_tool_arguments_fail_the_stream() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_invalid","content":[]}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_invalid","name":"lookup","input":{}}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let events = drain(
        anthropic(&mock)
            .language_model("claude-opus-5")
            .stream(tool_request("x"))
            .await
            .unwrap(),
    )
    .await;

    assert_terminal_contract(&events);
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::ToolCall(_)))
    );
    let Some(StreamEvent::Finish { finish, .. }) = events.last() else {
        panic!("missing finish");
    };
    assert_eq!(finish.reason, FinishReason::Error);
}

#[tokio::test]
async fn streamed_cyber_refusal_preserves_typed_stop_details() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_refusal","model":"claude-opus-5","content":[],"stop_reason":null,"usage":{"input_tokens":12,"output_tokens":0}}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal","category":"cyber","explanation":"The request could enable cyber harm."}}}"#,
        r#"{"type":"message_stop"}"#,
    ]);

    let events = drain(
        anthropic(&mock)
            .language_model("claude-opus-5")
            .stream(text_request("hi"))
            .await
            .expect("stream establishes"),
    )
    .await;

    assert_terminal_contract(&events);
    let metadata = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ProviderMetadata(metadata) => metadata.get("anthropic"),
            _ => None,
        })
        .expect("refusal metadata");
    assert_eq!(metadata["stop_details"]["category"], json!("cyber"));
    let Some(StreamEvent::Finish { finish, .. }) = events.last() else {
        panic!("missing finish");
    };
    assert_eq!(finish.reason, FinishReason::ContentFilter);
}

#[tokio::test]
async fn midstream_overloaded_error() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_7","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":1}}}"#,
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
    ]);
    let provider = anthropic(&mock);
    let events = drain(
        provider
            .language_model("claude-opus-5")
            .stream(text_request("hi"))
            .await
            .unwrap(),
    )
    .await;
    assert_terminal_contract(&events);
    let error = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::Error { error } => Some(error.clone()),
            _ => None,
        })
        .expect("error event");
    assert_eq!(error.kind(), ErrorKind::Overloaded);
    assert!(error.retryable());
}

#[tokio::test]
async fn midstream_error_aborts_open_provider_tool() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_abort","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":1}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_abort","name":"web_search","input":{}}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"par"}}"#,
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
    ]);
    let events = drain(
        anthropic(&mock)
            .language_model("claude-opus-5")
            .stream(text_request("weather?"))
            .await
            .unwrap(),
    )
    .await;

    assert_terminal_contract(&events);
    assert!(events.iter().any(|event| {
        matches!(event, StreamEvent::ProviderToolAbort { id } if id == "srvtoolu_abort")
    }));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::ProviderToolEnd(_))),
        "an aborted provider tool must not become replayable content"
    );
}

#[tokio::test]
async fn truncated_compaction_aborts_without_completed_part() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_compact_abort","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":150000,"output_tokens":1}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"compaction"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"Partial summary"}}"#,
    ]);
    let events = drain(
        anthropic(&mock)
            .language_model("claude-opus-5")
            .stream(text_request("continue"))
            .await
            .unwrap(),
    )
    .await;

    assert_terminal_contract(&events);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::CompactionAbort))
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::Compaction(_))),
        "an aborted compaction must not become replayable content"
    );
}

#[tokio::test]
async fn terminal_message_aborts_compaction_without_content_block_stop() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_compact_abort","content":[]}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"compaction"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"Partial summary"}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
        r#"{"type":"message_stop"}"#,
    ]);

    let events = drain(
        anthropic(&mock)
            .language_model("claude-opus-5")
            .stream(text_request("continue"))
            .await
            .unwrap(),
    )
    .await;

    assert_terminal_contract(&events);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::CompactionAbort))
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::Compaction(_)))
    );
}

#[tokio::test]
async fn terminal_message_aborts_provider_tool_without_content_block_stop() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_tool_abort","content":[]}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_abort","name":"web_search","input":{}}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"caido"}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
        r#"{"type":"message_stop"}"#,
    ]);

    let events = drain(
        anthropic(&mock)
            .language_model("claude-opus-5")
            .stream(text_request("search"))
            .await
            .unwrap(),
    )
    .await;

    assert_terminal_contract(&events);
    assert!(events.iter().any(|event| {
        matches!(event, StreamEvent::ProviderToolAbort { id } if id == "srvtoolu_abort")
    }));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::ProviderToolEnd(_)))
    );
}

#[tokio::test]
async fn server_tool_blocks_stream_and_replay() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_st","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":40,"output_tokens":1}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\": \"paris weather\"}"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://weather.example","title":"Paris weather"}]}}"#,
        r#"{"type":"content_block_stop","index":1}"#,
        r#"{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Sunny."}}"#,
        r#"{"type":"content_block_stop","index":2}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":9}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let provider = anthropic(&mock);
    let events = drain(
        provider
            .language_model("claude-opus-5")
            .stream(text_request("weather?"))
            .await
            .unwrap(),
    )
    .await;
    assert_terminal_contract(&events);

    let starts: Vec<(&str, &str)> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ProviderToolStart { id, kind } => Some((id.as_str(), kind.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(
        starts,
        vec![
            ("srvtoolu_1", "server_tool_use"),
            ("srvtoolu_1", "web_search_tool_result"),
        ]
    );
    let parts: Vec<llmwire::ProviderToolPart> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ProviderToolEnd(part) => Some(part.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].kind, "server_tool_use");
    assert_eq!(parts[0].namespace, "anthropic");
    assert_eq!(parts[0].payload["input"]["query"], "paris weather");
    assert_eq!(parts[1].payload["tool_use_id"], "srvtoolu_1");
    assert_eq!(
        parts[1].payload["content"][0]["url"],
        "https://weather.example"
    );

    let result = collect(&events).unwrap();
    let mock2 = MockTransport::shared();
    mock2.push_json(
        200,
        &json!({"id":"msg_ok","type":"message","role":"assistant","model":"claude-opus-5",
                "content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",
                "usage":{"input_tokens":1,"output_tokens":1}}),
    );
    let provider2 = anthropic(&mock2);
    let request = Request::builder()
        .message(Message::user("weather?"))
        .message(result.to_assistant_message())
        .message(Message::user("thanks"))
        .build();
    provider2
        .language_model("claude-opus-5")
        .generate(request)
        .await
        .expect("replay succeeds");
    let content = mock2.request_json(0)["messages"][1]["content"]
        .as_array()
        .unwrap()
        .clone();
    assert!(
        content
            .iter()
            .any(|block| block["type"] == "server_tool_use"
                && block["input"]["query"] == "paris weather")
    );
    assert!(
        content
            .iter()
            .any(|block| block["type"] == "web_search_tool_result"
                && block["tool_use_id"] == "srvtoolu_1")
    );
}

#[tokio::test]
async fn citations_stream_and_attach_to_text_metadata() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_cit","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Paris is the capital."}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","url":"https://example.com/fr","title":"France","cited_text":"Paris is the capital of France."}}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":6}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let provider = anthropic(&mock);
    let events = drain(
        provider
            .language_model("claude-opus-5")
            .stream(text_request("capital?"))
            .await
            .unwrap(),
    )
    .await;
    assert_terminal_contract(&events);

    let citation = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::Citation { text_id, citation } => {
                assert_eq!(text_id.as_deref(), Some("0"));
                Some(citation.clone())
            }
            _ => None,
        })
        .expect("citation event streams");
    assert_eq!(citation.url.as_deref(), Some("https://example.com/fr"));
    assert_eq!(citation.title.as_deref(), Some("France"));
    assert_eq!(
        citation.cited_text.as_deref(),
        Some("Paris is the capital of France.")
    );

    let metadata = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::TextEnd {
                provider_metadata, ..
            } => provider_metadata.get("anthropic").cloned(),
            _ => None,
        })
        .expect("text end carries citation metadata");
    assert_eq!(metadata["citations"][0]["url"], "https://example.com/fr");
}

#[tokio::test]
async fn midstream_errors_are_classified_like_http_errors() {
    for (error_type, message, kind) in [
        (
            "authentication_error",
            "invalid x-api-key",
            ErrorKind::Authentication,
        ),
        (
            "invalid_request_error",
            "prompt is too long: 250000 tokens",
            ErrorKind::ContextLength,
        ),
        ("not_found_error", "model: nope", ErrorKind::NotFound),
    ] {
        let mock = MockTransport::shared();
        mock.push_sse(&[&format!(
            r#"{{"type":"error","error":{{"type":"{error_type}","message":"{message}"}}}}"#
        )]);
        let events = drain(
            anthropic(&mock)
                .language_model("claude-opus-5")
                .stream(text_request("hi"))
                .await
                .unwrap(),
        )
        .await;

        assert_terminal_contract(&events);
        let error = events
            .iter()
            .find_map(|event| match event {
                StreamEvent::Error { error } => Some(error.clone()),
                _ => None,
            })
            .expect("error event");
        assert_eq!(error.kind(), kind, "{error_type}");
        assert_eq!(error.code(), Some(error_type));
    }
}

#[tokio::test]
async fn frames_with_the_wrong_shape_fail_the_stream() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_1","model":"claude-opus-5","content":[],"usage":{"input_tokens":1}}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":"many"}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let events = drain(
        anthropic(&mock)
            .language_model("claude-opus-5")
            .stream(text_request("hi"))
            .await
            .unwrap(),
    )
    .await;

    assert_terminal_contract(&events);
    let error = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::Error { error } => Some(error.clone()),
            _ => None,
        })
        .expect("a mismatched usage shape must not be dropped silently");
    assert_eq!(error.kind(), ErrorKind::MalformedResponse);
}

#[tokio::test]
async fn text_blocks_that_never_receive_deltas_are_not_kept() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_1","model":"claude-opus-5","content":[],"usage":{"input_tokens":1}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}"#,
        r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\":\"Paris\"}"}}"#,
        r#"{"type":"content_block_stop","index":1}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let result = anthropic(&mock)
        .language_model("claude-opus-5")
        .stream(tool_request("weather?"))
        .await
        .unwrap()
        .collect_result()
        .await
        .unwrap();

    assert!(
        !result
            .content
            .iter()
            .any(|part| matches!(part, AssistantPart::Text { text, .. } if text.is_empty())),
        "{:?}",
        result.content
    );
    assert_eq!(result.tool_calls().count(), 1);
}
