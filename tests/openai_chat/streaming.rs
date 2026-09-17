use super::*;

#[tokio::test]
async fn streaming_parallel_tool_calls_accumulate_by_index() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-5.6","choices":[{"index":0,"delta":{"role":"assistant","content":null},"finish_reason":null}]}"#,
        r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-5.6","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#,
        r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-5.6","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_y","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#,
        r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-5.6","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\":\"Paris\"}"}}]},"finish_reason":null}]}"#,
        r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-5.6","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"location\":\"London\"}"}}]},"finish_reason":null}]}"#,
        r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-5.6","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-5.6","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}"#,
        "[DONE]",
    ]);
    let provider = openai_chat(&mock);
    let events = drain(
        provider
            .language_model("gpt-5.6")
            .stream(tool_request("both cities"))
            .await
            .unwrap(),
    )
    .await;

    assert_terminal_contract(&events);
    let calls: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].call_id, "call_x");
    assert_eq!(calls[0].arguments, "{\"location\":\"Paris\"}");
    assert_eq!(calls[1].call_id, "call_y");
    assert_eq!(calls[1].arguments, "{\"location\":\"London\"}");

    let Some(StreamEvent::Finish { finish, usage }) = events.last() else {
        panic!()
    };
    assert_eq!(finish.reason, FinishReason::ToolCalls);
    assert_eq!(usage.input_tokens, Some(20));
}

#[tokio::test]
async fn response_metadata_emits_fields_that_arrive_in_later_chunks() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":null}]}"#,
        r#"{"model":"gpt-5.6-sol","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}"#,
        r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ]);

    let events = drain(
        openai_chat(&mock)
            .language_model("gpt-5.6-sol")
            .stream(text_request("hi"))
            .await
            .unwrap(),
    )
    .await;

    let metadata: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ResponseMetadata(metadata) => Some(metadata),
            _ => None,
        })
        .collect();
    assert_eq!(metadata.len(), 2);
    assert_eq!(metadata[0].id.as_deref(), Some("chatcmpl-1"));
    assert_eq!(metadata[0].model, None);
    assert_eq!(metadata[1].id, None);
    assert_eq!(metadata[1].model.as_deref(), Some("gpt-5.6-sol"));
}

#[tokio::test]
async fn late_name_fragments_are_buffered() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"c2","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":"}}]},"finish_reason":null}]}"#,
        r#"{"id":"c2","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_z","type":"function","function":{"name":"tool_z","arguments":"1}"}}]},"finish_reason":null}]}"#,
        r#"{"id":"c2","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        "[DONE]",
    ]);
    let provider = openai_chat(&mock);
    let events = drain(
        provider
            .language_model("some-model")
            .stream(tool_request("x"))
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
    assert_eq!(call.name, "tool_z");
    assert_eq!(call.arguments, "{\"a\":1}");
}

#[tokio::test]
async fn eof_with_finish_reason_but_no_done_completes() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"c3","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#,
        r#"{"id":"c3","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
    ]);
    let provider = openai_chat(&mock);
    let events = drain(
        provider
            .language_model("m")
            .stream(text_request("hi"))
            .await
            .unwrap(),
    )
    .await;
    assert_terminal_contract(&events);
    let Some(StreamEvent::Finish { finish, .. }) = events.last() else {
        panic!()
    };
    assert_eq!(finish.reason, FinishReason::Stop);
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::Error { .. }))
    );
}

#[tokio::test]
async fn eof_without_finish_reason_is_truncation() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"c4","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"par"},"finish_reason":null}]}"#,
    ]);
    let provider = openai_chat(&mock);
    let events = drain(
        provider
            .language_model("m")
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
}

#[tokio::test]
async fn reasoning_content_deltas_precede_text() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"c5","object":"chat.completion.chunk","created":1,"model":"deepseek-r1","choices":[{"index":0,"delta":{"reasoning_content":"Let me think. "},"finish_reason":null}]}"#,
        r#"{"id":"c5","object":"chat.completion.chunk","created":1,"model":"deepseek-r1","choices":[{"index":0,"delta":{"reasoning_content":"Done."},"finish_reason":null}]}"#,
        r#"{"id":"c5","object":"chat.completion.chunk","created":1,"model":"deepseek-r1","choices":[{"index":0,"delta":{"content":"Answer"},"finish_reason":null}]}"#,
        r#"{"id":"c5","object":"chat.completion.chunk","created":1,"model":"deepseek-r1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ]);
    let provider = openai_chat(&mock);
    let stream = provider
        .language_model("deepseek-r1")
        .stream(text_request("hi"))
        .await
        .unwrap();
    let result = stream.collect_result().await.expect("collect");
    let reasoning: Vec<_> = result.reasoning().collect();
    assert_eq!(reasoning.len(), 1);
    assert_eq!(reasoning[0].visible_text(), "Let me think. Done.");
    assert_eq!(result.text(), "Answer");
    assert!(matches!(
        result.content[0],
        cogfer::AssistantPart::Reasoning(_)
    ));
}

#[tokio::test]
async fn late_tool_call_id_is_not_replaced_by_a_synthetic_one() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"c7","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"get_weather"}}]},"finish_reason":null}]}"#,
        r#"{"id":"c7","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_real","function":{"arguments":"{\"a\":1}"}}]},"finish_reason":null}]}"#,
        r#"{"id":"c7","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        "[DONE]",
    ]);
    let provider = openai_chat(&mock);
    let events = drain(
        provider
            .language_model("m")
            .stream(tool_request("x"))
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
    assert_eq!(call.call_id, "call_real");
    assert_eq!(call.name, "get_weather");
    assert_eq!(call.arguments, "{\"a\":1}");
}

#[tokio::test]
async fn extra_choices_do_not_corrupt_the_first() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"c8","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"first"},"finish_reason":null},{"index":1,"delta":{"content":"second"},"finish_reason":null}]}"#,
        r#"{"id":"c8","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ]);
    let provider = openai_chat(&mock);
    let result = provider
        .language_model("m")
        .stream(text_request("hi"))
        .await
        .unwrap()
        .collect_result()
        .await
        .expect("collect");
    assert_eq!(result.text(), "first");
}

#[tokio::test]
async fn unindexed_tool_calls_with_distinct_ids_stay_separate() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"c9","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_a","function":{"name":"foo","arguments":"{\"a\":1}"}}]},"finish_reason":null}]}"#,
        r#"{"id":"c9","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_b","function":{"name":"bar","arguments":"{\"b\":2}"}}]},"finish_reason":null}]}"#,
        r#"{"id":"c9","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        "[DONE]",
    ]);
    let provider = openai_chat(&mock);
    let result = provider
        .language_model("m")
        .stream(tool_request("hi"))
        .await
        .unwrap()
        .collect_result()
        .await
        .expect("collect");
    let calls: Vec<&ToolCall> = result.tool_calls().collect();
    assert_eq!(calls.len(), 2, "{calls:?}");
    assert_eq!(calls[0].call_id, "call_a");
    assert_eq!(calls[0].name, "foo");
    assert_eq!(calls[0].arguments, "{\"a\":1}");
    assert_eq!(calls[1].call_id, "call_b");
    assert_eq!(calls[1].name, "bar");
    assert_eq!(calls[1].arguments, "{\"b\":2}");
}

#[tokio::test]
async fn error_finish_reason_fails_the_stream() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"c10","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"foo","arguments":"{\"a\":"}}]},"finish_reason":null}]}"#,
        r#"{"id":"c10","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"error"}]}"#,
        "[DONE]",
    ]);
    let provider = openai_chat(&mock);
    let events = drain(
        provider
            .language_model("m")
            .stream(tool_request("hi"))
            .await
            .unwrap(),
    )
    .await;
    assert_terminal_contract(&events);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::Error { .. }))
    );
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
async fn length_finish_does_not_promote_a_complete_looking_tool_call() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"c_length","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_partial","function":{"name":"delete","arguments":"{\"path\":\"/tmp\"}"}}]},"finish_reason":null}]}"#,
        r#"{"id":"c_length","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#,
        "[DONE]",
    ]);
    let events = drain(
        openai_chat(&mock)
            .language_model("m")
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
async fn stop_finish_still_emits_completed_tool_calls() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"c_forced","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_forced","function":{"name":"get_weather","arguments":"{\"location\":"}}]},"finish_reason":null}]}"#,
        r#"{"id":"c_forced","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]},"finish_reason":null}]}"#,
        r#"{"id":"c_forced","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ]);
    let events = drain(
        openai_chat(&mock)
            .language_model("m")
            .stream(tool_request("x"))
            .await
            .unwrap(),
    )
    .await;

    assert_terminal_contract(&events);
    let call = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ToolCall(call) => Some(call),
            _ => None,
        })
        .expect("completed call is emitted despite finish_reason stop");
    assert_eq!(call.call_id, "call_forced");
    assert_eq!(call.arguments, "{\"location\":\"Paris\"}");
    let Some(StreamEvent::Finish { finish, .. }) = events.last() else {
        panic!("missing finish");
    };
    assert_eq!(finish.reason, FinishReason::ToolCalls);
    assert_eq!(finish.raw.as_deref(), Some("stop"));
}

#[tokio::test]
async fn invalid_completed_tool_arguments_fail_the_stream() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"c_invalid","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_invalid","function":{"name":"lookup","arguments":"{\"q\":"}}]},"finish_reason":null}]}"#,
        r#"{"id":"c_invalid","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        "[DONE]",
    ]);
    let events = drain(
        openai_chat(&mock)
            .language_model("m")
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
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::Error { .. }))
    );
    let Some(StreamEvent::Finish { finish, .. }) = events.last() else {
        panic!("missing finish");
    };
    assert_eq!(finish.reason, FinishReason::Error);
}

#[tokio::test]
async fn streamed_refusal_carries_marker() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"c11","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"reasoning_content":"hmm"},"finish_reason":null}]}"#,
        r#"{"id":"c11","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"refusal":"I can't help with that."},"finish_reason":null}]}"#,
        r#"{"id":"c11","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ]);
    let provider = openai_chat(&mock);
    let result = provider
        .language_model("m")
        .stream(text_request("hi"))
        .await
        .unwrap()
        .collect_result()
        .await
        .expect("collect");
    assert_eq!(result.text(), "I can't help with that.");
    assert_eq!(result.finish.reason, FinishReason::ContentFilter);
    let marker = result
        .content
        .iter()
        .find_map(|part| match part {
            cogfer::AssistantPart::Text {
                provider_metadata, ..
            } => provider_metadata
                .get("openai")
                .and_then(|namespace| namespace.get("content_type"))
                .cloned(),
            _ => None,
        })
        .expect("refusal marker");
    assert_eq!(marker, json!("refusal"));
    assert_eq!(result.reasoning().count(), 1);
}

#[tokio::test]
async fn type_mismatched_chunk_fails_the_stream() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"c12","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":5},"finish_reason":null}]}"#,
        "[DONE]",
    ]);
    let provider = openai_chat(&mock);
    let error = provider
        .language_model("m")
        .stream(text_request("hi"))
        .await
        .unwrap()
        .collect_result()
        .await
        .expect_err("malformed chunk fails");
    assert_eq!(error.kind(), ErrorKind::MalformedResponse);
}

#[tokio::test]
async fn chat_annotations_stream_as_citations_and_persist() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"chat_ann","model":"gpt-5.6","choices":[{"index":0,"delta":{"role":"assistant","content":"Paris"}}]}"#,
        r#"{"id":"chat_ann","model":"gpt-5.6","choices":[{"index":0,"delta":{"content":" is the capital.","annotations":[{"type":"url_citation","url_citation":{"url":"https://example.com/paris","title":"Paris"},"url":"https://example.com/paris","title":"Paris"}]}}]}"#,
        r#"{"id":"chat_ann","model":"gpt-5.6","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":5,"total_tokens":10}}"#,
        "[DONE]",
    ]);
    let provider = openai_chat(&mock);
    let events = drain(
        provider
            .language_model("gpt-5.6")
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
                assert_eq!(text_id.as_deref(), Some("t0"));
                Some(citation.clone())
            }
            _ => None,
        })
        .expect("citation event streams");
    assert_eq!(citation.url.as_deref(), Some("https://example.com/paris"));
    let metadata = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::TextEnd {
                provider_metadata, ..
            } => provider_metadata.get("openai").cloned(),
            _ => None,
        })
        .expect("text end carries annotations");
    assert_eq!(
        metadata["annotations"][0]["url"],
        "https://example.com/paris"
    );
}

#[tokio::test]
async fn chat_refusal_identity_rides_on_text_start() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"chat_ref","model":"gpt-5.6","choices":[{"index":0,"delta":{"role":"assistant","refusal":"I can't "}}]}"#,
        r#"{"id":"chat_ref","model":"gpt-5.6","choices":[{"index":0,"delta":{"refusal":"help with that."}}]}"#,
        r#"{"id":"chat_ref","model":"gpt-5.6","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ]);
    let provider = openai_chat(&mock);
    let events = drain(
        provider
            .language_model("gpt-5.6")
            .stream(text_request("hi"))
            .await
            .unwrap(),
    )
    .await;
    let start = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::TextStart {
                provider_metadata, ..
            } => Some(provider_metadata.clone()),
            _ => None,
        })
        .expect("text start present");
    assert_eq!(
        start.get("openai").and_then(|ns| ns.get("content_type")),
        Some(&serde_json::json!("refusal")),
        "refusal must be identifiable when the block opens"
    );
}

#[tokio::test]
async fn json_body_on_a_stream_request_is_replayed_as_events() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "chatcmpl-buffered", "model": "gpt-5.6",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "buffered"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        }),
    );
    let events = drain(
        openai_chat(&mock)
            .language_model("gpt-5.6")
            .stream(text_request("hi"))
            .await
            .unwrap(),
    )
    .await;

    assert_terminal_contract(&events);
    assert_eq!(
        kinds(&events),
        vec![
            "stream-start",
            "response-metadata",
            "text-start",
            "text-delta",
            "text-end",
            "finish",
        ]
    );
    let Some(StreamEvent::Finish { finish, usage }) = events.last() else {
        panic!("missing finish");
    };
    assert_eq!(finish.reason, FinishReason::Stop);
    assert_eq!(usage.input_tokens, Some(3));
}

#[tokio::test]
async fn json_error_body_on_a_stream_request_fails_with_the_provider_message() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({"error": {"message": "model overloaded", "code": 503}}),
    );
    let error = openai_chat(&mock)
        .language_model("gpt-5.6")
        .stream(text_request("hi"))
        .await
        .expect_err("a JSON error body is not a stream");

    assert_eq!(error.kind(), ErrorKind::Overloaded);
    assert_eq!(error.message(), "model overloaded");
}

#[tokio::test]
async fn transport_failure_mid_stream_is_not_retryable() {
    let mock = MockTransport::shared();
    mock.push_stream_then_error(
        vec![bytes::Bytes::from_static(
            b"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n",
        )],
        ErrorKind::Timeout,
    );
    let events = drain(
        openai_chat(&mock)
            .language_model("gpt-5.6")
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
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(!error.retryable());
    assert!(
        events.iter().any(
            |event| matches!(event, StreamEvent::TextDelta { delta, .. } if delta == "partial")
        )
    );
}

#[tokio::test]
async fn stream_chunk_with_the_wrong_shape_fails_the_stream() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"c1","choices":[{"index":0,"delta":{"content":"ok"}}]}"#,
        r#"{"id":"c1","choices":"not-an-array"}"#,
    ]);
    let events = drain(
        openai_chat(&mock)
            .language_model("gpt-5.6")
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
    assert_eq!(error.kind(), ErrorKind::MalformedResponse);
}
