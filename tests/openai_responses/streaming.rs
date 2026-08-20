use bytes::Bytes;

use super::*;

#[tokio::test]
async fn streaming_lifecycle_with_authoritative_done_arguments() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_5","model":"gpt-5.6","status":"in_progress"},"sequence_number":0}"#,
        r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_5","type":"reasoning","summary":[]},"sequence_number":1}"#,
        r#"{"type":"response.reasoning_summary_text.delta","item_id":"rs_5","output_index":0,"summary_index":0,"delta":"Thinking...","sequence_number":2}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_5","type":"reasoning","summary":[{"type":"summary_text","text":"Thinking..."}],"encrypted_content":"ENC5"},"sequence_number":3}"#,
        r#"{"type":"response.output_item.added","output_index":1,"item":{"id":"msg_5","type":"message","role":"assistant","status":"in_progress","content":[]},"sequence_number":4}"#,
        r#"{"type":"response.content_part.added","item_id":"msg_5","output_index":1,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]},"sequence_number":5}"#,
        r#"{"type":"response.output_text.delta","item_id":"msg_5","output_index":1,"content_index":0,"delta":"Let me check","sequence_number":6,"obfuscation":"xK9"}"#,
        r#"{"type":"response.content_part.done","item_id":"msg_5","output_index":1,"content_index":0,"part":{"type":"output_text","text":"Let me check","annotations":[]},"sequence_number":7}"#,
        r#"{"type":"response.output_item.done","output_index":1,"item":{"id":"msg_5","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Let me check","annotations":[]}]},"sequence_number":8}"#,
        r#"{"type":"response.output_item.added","output_index":2,"item":{"id":"fc_5","type":"function_call","call_id":"call_5","name":"get_weather","arguments":"","status":"in_progress"},"sequence_number":8}"#,
        r#"{"type":"response.function_call_arguments.delta","item_id":"fc_5","output_index":2,"delta":"{\"loc","sequence_number":9}"#,
        r#"{"type":"response.function_call_arguments.delta","item_id":"fc_5","output_index":2,"delta":"ation\":\"Par","sequence_number":10}"#,
        r#"{"type":"response.output_item.done","output_index":2,"item":{"id":"fc_5","type":"function_call","call_id":"call_5","name":"get_weather","arguments":"{\"location\":\"Paris\"}","status":"completed"},"sequence_number":11}"#,
        r#"{"type":"response.completed","response":{"id":"resp_5","status":"completed","output":[],"usage":{"input_tokens":50,"output_tokens":20,"total_tokens":70,"output_tokens_details":{"reasoning_tokens":8}}},"sequence_number":12}"#,
    ]);
    let provider = openai_responses(&mock);
    let events = drain(
        provider
            .language_model("gpt-5.6")
            .stream(tool_request("weather?"))
            .await
            .expect("stream establishes"),
    )
    .await;

    assert_terminal_contract(&events);
    assert_eq!(
        kinds(&events),
        vec![
            "stream-start",
            "provider-metadata",
            "response-metadata",
            "reasoning-start",
            "reasoning-delta",
            "reasoning-end",
            "text-start",
            "text-delta",
            "text-end",
            "tool-input-start",
            "tool-input-delta",
            "tool-input-delta",
            "tool-input-end",
            "tool-call",
            "finish",
        ]
    );

    let call = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .expect("tool call event");
    assert_eq!(call.arguments, "{\"location\":\"Paris\"}");
    assert_eq!(call.call_id, "call_5");
    assert_eq!(call.item_id.as_deref(), Some("fc_5"));

    let reasoning = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ReasoningEnd { part, .. } => Some(part.clone()),
            _ => None,
        })
        .expect("reasoning end");
    assert!(
        reasoning.content.iter().any(
            |content| matches!(content, ReasoningContent::Encrypted { data } if data == "ENC5")
        )
    );

    let Some(StreamEvent::Finish { finish, usage }) = events.last() else {
        panic!("missing finish");
    };
    assert_eq!(finish.reason, FinishReason::ToolCalls);
    assert_eq!(usage.reasoning_tokens, Some(8));
}

#[tokio::test]
async fn stream_error_event_then_failed_is_single_error() {
    let mock = MockTransport::shared();
    let frames = [
        r#"{"type":"response.created","response":{"id":"resp_6","model":"gpt-5.6-sol"},"sequence_number":0}"#,
        r#"{"type":"error","code":"server_error","message":"exploded","param":null,"sequence_number":1}"#,
        r#"{"type":"response.failed","response":{"id":"resp_6","status":"failed","error":{"code":"server_error","message":"exploded"},"output":[]},"sequence_number":2}"#,
    ];
    mock.push_stream_chunks(
        200,
        vec![
            ("content-type".into(), "text/event-stream".into()),
            ("x-request-id".into(), "req_failed_56".into()),
        ],
        frames
            .iter()
            .map(|frame| Bytes::from(format!("data: {frame}\n\n")))
            .collect(),
    );
    let provider = openai_responses(&mock);
    let events = drain(
        provider
            .language_model("gpt-5.6-sol")
            .stream(text_request("hi"))
            .await
            .unwrap(),
    )
    .await;
    assert_terminal_contract(&events);
    let errors = events
        .iter()
        .filter(|event| matches!(event, StreamEvent::Error { .. }))
        .count();
    assert_eq!(errors, 1, "error+failed pair dedupes to one error event");
    let error = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::Error { error } => Some(error),
            _ => None,
        })
        .expect("provider error");
    assert_eq!(error.model(), Some("gpt-5.6-sol"));
    assert_eq!(error.request_id(), Some("req_failed_56"));
    let Some(StreamEvent::Finish { finish, .. }) = events.last() else {
        panic!()
    };
    assert_eq!(finish.reason, FinishReason::Error);
}

#[tokio::test]
async fn streamed_content_policy_error_is_typed() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_policy","model":"gpt-5.6-sol"}}"#,
        r#"{"type":"error","code":"content_policy_violation","message":"Blocked by policy"}"#,
        r#"{"type":"response.failed","response":{"id":"resp_policy","model":"gpt-5.6-sol","status":"failed","error":{"code":"content_policy_violation","message":"Blocked by policy"},"output":[]}}"#,
    ]);

    let events = drain(
        openai_responses(&mock)
            .language_model("gpt-5.6-sol")
            .stream(text_request("hi"))
            .await
            .expect("stream establishes"),
    )
    .await;

    let error = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::Error { error } => Some(error),
            _ => None,
        })
        .expect("policy error");
    assert_eq!(error.kind(), ErrorKind::ContentPolicy);
    assert_eq!(error.code(), Some("content_policy_violation"));
    assert_terminal_contract(&events);
}

#[tokio::test]
async fn failed_response_aborts_open_provider_tool() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_tool_abort","model":"gpt-5.6"}}"#,
        r#"{"type":"response.output_item.added","item":{"id":"ws_abort","type":"web_search_call","status":"in_progress"}}"#,
        r#"{"type":"response.web_search_call.searching","item_id":"ws_abort"}"#,
        r#"{"type":"response.failed","response":{"id":"resp_tool_abort","status":"failed","error":{"code":"server_error","message":"search failed"},"output":[]}}"#,
    ]);
    let events = drain(
        openai_responses(&mock)
            .language_model("gpt-5.6")
            .stream(text_request("search"))
            .await
            .unwrap(),
    )
    .await;

    assert_terminal_contract(&events);
    assert!(events.iter().any(|event| {
        matches!(event, StreamEvent::ProviderToolAbort { id } if id == "ws_abort")
    }));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::ProviderToolEnd(_))),
        "an aborted provider tool must not become replayable content"
    );
}

#[tokio::test]
async fn error_then_eof_aborts_compaction_and_dedupes_error() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_compact_abort","model":"gpt-5.6"}}"#,
        r#"{"type":"response.output_item.added","item":{"id":"cmp_abort","type":"compaction","status":"in_progress"}}"#,
        r#"{"type":"error","code":"server_error","message":"compaction failed"}"#,
    ]);
    let events = drain(
        openai_responses(&mock)
            .language_model("gpt-5.6")
            .stream(text_request("continue"))
            .await
            .unwrap(),
    )
    .await;

    assert_terminal_contract(&events);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamEvent::Error { .. }))
            .count(),
        1
    );
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
async fn incomplete_response_does_not_promote_a_completed_function_item() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.output_item.added","item":{"id":"fc_partial","type":"function_call","call_id":"call_partial","name":"delete","arguments":"","status":"in_progress"}}"#,
        r#"{"type":"response.function_call_arguments.delta","item_id":"fc_partial","delta":"{\"path\":\"/tmp\"}"}"#,
        r#"{"type":"response.output_item.done","item":{"id":"fc_partial","type":"function_call","call_id":"call_partial","name":"delete","arguments":"{\"path\":\"/tmp\"}","status":"completed"}}"#,
        r#"{"type":"response.incomplete","response":{"id":"resp_partial","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[]}}"#,
    ]);
    let events = drain(
        openai_responses(&mock)
            .language_model("gpt-5.6")
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
async fn incomplete_function_item_is_not_promoted_by_a_completed_response() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.output_item.added","item":{"id":"fc_partial","type":"function_call","call_id":"call_partial","name":"delete","arguments":"","status":"in_progress"}}"#,
        r#"{"type":"response.function_call_arguments.delta","item_id":"fc_partial","delta":"{\"path\":\"/tmp\"}"}"#,
        r#"{"type":"response.output_item.done","item":{"id":"fc_partial","type":"function_call","call_id":"call_partial","name":"delete","arguments":"{\"path\":\"/tmp\"}","status":"incomplete"}}"#,
        r#"{"type":"response.completed","response":{"id":"resp_partial","status":"completed","output":[]}}"#,
    ]);
    let events = drain(
        openai_responses(&mock)
            .language_model("gpt-5.6")
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
    assert_eq!(finish.reason, FinishReason::Stop);
}

#[tokio::test]
async fn truncated_stream_reports_error_finish() {
    let mock = MockTransport::shared();
    let frames = [
        r#"{"type":"response.created","response":{"id":"resp_7","model":"gpt-5.6-sol"},"sequence_number":0}"#,
        r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_7","type":"message","role":"assistant","status":"in_progress","content":[]},"sequence_number":1}"#,
        r#"{"type":"response.output_text.delta","item_id":"msg_7","output_index":0,"content_index":0,"delta":"par","sequence_number":2}"#,
    ];
    mock.push_stream_chunks(
        200,
        vec![
            ("content-type".into(), "text/event-stream".into()),
            ("x-request-id".into(), "req_safeguard_56".into()),
        ],
        frames
            .iter()
            .map(|frame| Bytes::from(format!("data: {frame}\n\n")))
            .collect(),
    );
    let provider = openai_responses(&mock);
    let events = drain(
        provider
            .language_model("gpt-5.6-sol")
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
        .expect("truncation error");
    assert_eq!(error.kind(), ErrorKind::TruncatedStream);
    assert_eq!(error.origin(), Some("openai-responses"));
    assert_eq!(error.model(), Some("gpt-5.6-sol"));
    assert_eq!(error.request_id(), Some("req_safeguard_56"));
    assert!(!error.retryable());
}

#[tokio::test]
async fn unknown_events_surface_as_raw_only_when_requested() {
    let frames = [
        r#"{"type":"response.created","response":{"id":"resp_8","model":"gpt-5.6"},"sequence_number":0}"#,
        r#"{"type":"response.brand_new_event","something":"else","sequence_number":1}"#,
        r#"{"type":"response.completed","response":{"id":"resp_8","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}},"sequence_number":2}"#,
    ];

    let mock = MockTransport::shared();
    mock.push_sse(&frames);
    let provider = openai_responses(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .include_raw_events(true)
        .build();
    let events = drain(
        provider
            .language_model("gpt-5.6")
            .stream(request)
            .await
            .unwrap(),
    )
    .await;
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::Raw { .. }))
    );

    let mock = MockTransport::shared();
    mock.push_sse(&frames);
    let provider = openai_responses(&mock);
    let events = drain(
        provider
            .language_model("gpt-5.6")
            .stream(text_request("hi"))
            .await
            .unwrap(),
    )
    .await;
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::Raw { .. }))
    );
}

#[tokio::test]
async fn streaming_preserves_phase_and_refusal_metadata() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_ph","model":"gpt-5.6","status":"in_progress"},"sequence_number":0}"#,
        r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_ph","type":"message","role":"assistant","status":"in_progress","content":[]},"sequence_number":1}"#,
        r#"{"type":"response.content_part.added","item_id":"msg_ph","output_index":0,"content_index":0,"part":{"type":"refusal","refusal":""},"sequence_number":2}"#,
        r#"{"type":"response.refusal.delta","item_id":"msg_ph","output_index":0,"content_index":0,"delta":"I cannot do that.","sequence_number":3}"#,
        r#"{"type":"response.content_part.done","item_id":"msg_ph","output_index":0,"content_index":0,"part":{"type":"refusal","refusal":"I cannot do that."},"sequence_number":4}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_ph","type":"message","role":"assistant","status":"completed","phase":"final_answer","content":[{"type":"refusal","refusal":"I cannot do that."}]},"sequence_number":5}"#,
        r#"{"type":"response.completed","response":{"id":"resp_ph","status":"completed","output":[],"usage":{"input_tokens":5,"output_tokens":4,"total_tokens":9}},"sequence_number":6}"#,
    ]);
    let provider = openai_responses(&mock);
    let events = drain(
        provider
            .language_model("gpt-5.6")
            .stream(text_request("hi"))
            .await
            .expect("stream establishes"),
    )
    .await;
    assert_terminal_contract(&events);

    let metadata = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::TextEnd {
                provider_metadata, ..
            } => Some(provider_metadata.clone()),
            _ => None,
        })
        .expect("text end");
    let openai = metadata.get("openai").expect("openai namespace");
    assert_eq!(openai["phase"], serde_json::json!("final_answer"));
    assert_eq!(openai["content_type"], serde_json::json!("refusal"));

    let result = collect(&events).expect("result");
    assert_eq!(result.finish.reason, FinishReason::ContentFilter);
    let caido_ai::AssistantPart::Text {
        provider_metadata, ..
    } = &result.content[0]
    else {
        panic!("expected text part");
    };
    let openai = provider_metadata.get("openai").expect("openai namespace");
    assert_eq!(openai["content_type"], serde_json::json!("refusal"));
}

#[tokio::test]
async fn server_tool_items_stream_with_lifecycle_and_replay() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_ws","model":"gpt-5.6","status":"in_progress"},"sequence_number":0}"#,
        r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"ws_1","type":"web_search_call","status":"in_progress"},"sequence_number":1}"#,
        r#"{"type":"response.web_search_call.in_progress","item_id":"ws_1","output_index":0,"sequence_number":2}"#,
        r#"{"type":"response.web_search_call.searching","item_id":"ws_1","output_index":0,"sequence_number":3}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"weather paris"}},"sequence_number":4}"#,
        r#"{"type":"response.output_item.added","output_index":1,"item":{"id":"msg_ws","type":"message","role":"assistant","status":"in_progress","content":[]},"sequence_number":5}"#,
        r#"{"type":"response.content_part.added","item_id":"msg_ws","output_index":1,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]},"sequence_number":6}"#,
        r#"{"type":"response.output_text.delta","item_id":"msg_ws","output_index":1,"content_index":0,"delta":"Sunny.","sequence_number":7}"#,
        r#"{"type":"response.output_item.done","output_index":1,"item":{"id":"msg_ws","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Sunny.","annotations":[]}]},"sequence_number":8}"#,
        r#"{"type":"response.completed","response":{"id":"resp_ws","status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}},"sequence_number":9}"#,
    ]);
    let provider = openai_responses(&mock);
    let events = drain(
        provider
            .language_model("gpt-5.6")
            .stream(text_request("weather?"))
            .await
            .expect("stream establishes"),
    )
    .await;
    assert_terminal_contract(&events);
    assert_eq!(
        kinds(&events),
        vec![
            "stream-start",
            "provider-metadata",
            "response-metadata",
            "provider-tool-start",
            "provider-tool-update",
            "provider-tool-update",
            "provider-tool-end",
            "text-start",
            "text-delta",
            "text-end",
            "finish",
        ]
    );
    let statuses: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ProviderToolUpdate { id, status } => {
                assert_eq!(id, "ws_1");
                Some(status.as_str())
            }
            _ => None,
        })
        .collect();
    assert_eq!(statuses, vec!["in_progress", "searching"]);

    let result = collect(&events).expect("result accumulates");
    let tool = result
        .content
        .iter()
        .find_map(|part| match part {
            caido_ai::AssistantPart::ProviderTool {
                provider_tool: tool,
            } => Some(tool),
            _ => None,
        })
        .expect("server tool part lands in content");
    assert_eq!(tool.kind, "web_search_call");
    assert_eq!(tool.namespace, "openai");
    assert_eq!(tool.payload["action"]["query"], "weather paris");

    let mock2 = MockTransport::shared();
    mock2.push_json(200, &minimal_completed());
    let provider2 = openai_responses(&mock2);
    let request = Request::builder()
        .message(Message::user("weather?"))
        .message(result.to_assistant_message())
        .message(Message::user("thanks"))
        .build();
    provider2
        .language_model("gpt-5.6")
        .generate(request)
        .await
        .expect("replay succeeds");
    let input = mock2.request_json(0)["input"].as_array().unwrap().clone();
    let replayed = input
        .iter()
        .find(|item| item["type"] == "web_search_call")
        .expect("server tool item replayed");
    assert_eq!(replayed["id"], "ws_1");
    assert_eq!(replayed["action"]["query"], "weather paris");
}

#[tokio::test]
async fn text_annotations_stream_as_citations_and_persist() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_ann","model":"gpt-5.6","status":"in_progress"},"sequence_number":0}"#,
        r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_a","type":"message","role":"assistant","status":"in_progress","content":[]},"sequence_number":1}"#,
        r#"{"type":"response.content_part.added","item_id":"msg_a","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]},"sequence_number":2}"#,
        r#"{"type":"response.output_text.delta","item_id":"msg_a","output_index":0,"content_index":0,"delta":"Paris is the capital.","sequence_number":3}"#,
        r#"{"type":"response.output_text.annotation.added","item_id":"msg_a","output_index":0,"content_index":0,"annotation_index":0,"annotation":{"type":"url_citation","url":"https://example.com/paris","title":"Paris","start_index":0,"end_index":5},"sequence_number":4}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_a","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Paris is the capital.","annotations":[{"type":"url_citation","url":"https://example.com/paris","title":"Paris","start_index":0,"end_index":5}]}]},"sequence_number":5}"#,
        r#"{"type":"response.completed","response":{"id":"resp_ann","status":"completed","output":[],"usage":{"input_tokens":5,"output_tokens":5,"total_tokens":10}},"sequence_number":6}"#,
    ]);
    let provider = openai_responses(&mock);
    let events = drain(
        provider
            .language_model("gpt-5.6")
            .stream(text_request("capital of France?"))
            .await
            .expect("stream establishes"),
    )
    .await;
    assert_terminal_contract(&events);

    let (text_id, citation) = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::Citation { text_id, citation } => {
                Some((text_id.clone(), citation.clone()))
            }
            _ => None,
        })
        .expect("citation event streams");
    assert_eq!(text_id.as_deref(), Some("msg_a:0"));
    assert_eq!(citation.url.as_deref(), Some("https://example.com/paris"));
    assert_eq!(citation.title.as_deref(), Some("Paris"));
    assert_eq!(citation.raw["start_index"], 0);

    let annotations = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::TextEnd {
                provider_metadata, ..
            } => provider_metadata.get("openai").cloned(),
            _ => None,
        })
        .expect("text end carries metadata");
    assert_eq!(
        annotations["annotations"][0]["url"],
        "https://example.com/paris"
    );
}

#[tokio::test]
async fn refusal_identity_rides_on_text_start() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_ref","model":"gpt-5.6","status":"in_progress"},"sequence_number":0}"#,
        r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_ref","type":"message","role":"assistant","status":"in_progress","content":[]},"sequence_number":1}"#,
        r#"{"type":"response.content_part.added","item_id":"msg_ref","output_index":0,"content_index":0,"part":{"type":"refusal","refusal":""},"sequence_number":2}"#,
        r#"{"type":"response.refusal.delta","item_id":"msg_ref","output_index":0,"content_index":0,"delta":"I can't help with that.","sequence_number":3}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_ref","type":"message","role":"assistant","status":"completed","content":[{"type":"refusal","refusal":"I can't help with that."}]},"sequence_number":4}"#,
        r#"{"type":"response.completed","response":{"id":"resp_ref","status":"completed","output":[],"usage":{"input_tokens":5,"output_tokens":5,"total_tokens":10}},"sequence_number":5}"#,
    ]);
    let provider = openai_responses(&mock);
    let events = drain(
        provider
            .language_model("gpt-5.6")
            .stream(text_request("hi"))
            .await
            .expect("stream establishes"),
    )
    .await;
    let start_metadata = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::TextStart {
                provider_metadata, ..
            } => Some(provider_metadata.clone()),
            _ => None,
        })
        .expect("text start present");
    assert_eq!(
        start_metadata
            .get("openai")
            .and_then(|ns| ns.get("content_type")),
        Some(&json!("refusal")),
        "refusal must be identifiable when the block opens"
    );
}

#[tokio::test]
async fn streamed_turns_record_their_origin_profile() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_9","model":"gpt-5.6","status":"in_progress"}}"#,
        r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_9","type":"message","role":"assistant","status":"in_progress","content":[]}}"#,
        r#"{"type":"response.output_text.delta","item_id":"msg_9","output_index":0,"content_index":0,"delta":"hey"}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_9","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hey","annotations":[]}]}}"#,
        r#"{"type":"response.completed","response":{"id":"resp_9","status":"completed","output":[],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}"#,
    ]);
    let provider = openai_responses(&mock);

    let events = drain(
        provider
            .language_model("gpt-5.6")
            .stream(text_request("hi"))
            .await
            .expect("stream establishes"),
    )
    .await;

    let result = collect(&events).expect("stream accumulates");
    assert_eq!(
        result.provider_metadata.get("caido-ai").unwrap()["profile"],
        "openai-responses"
    );
}
