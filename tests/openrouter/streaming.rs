use super::*;

#[tokio::test]
async fn streamed_response_preserves_cost_and_routing_metadata() {
    let mock = MockTransport::shared();
    let usage = json!({
        "prompt_tokens": 82,
        "completion_tokens": 75,
        "total_tokens": 157,
        "cost": 0.000457,
        "is_byok": false,
        "cost_details": {"upstream_inference_cost": 0.000457}
    });
    mock.push_sse(&[
        r#"{"id":"gen-stream-metadata","model":"anthropic/claude-haiku-4.5","provider":"Amazon Bedrock","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}"#,
        &json!({
            "id": "gen-stream-metadata",
            "model": "anthropic/claude-haiku-4.5",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": usage
        })
        .to_string(),
        "[DONE]",
    ]);
    let result = openrouter(&mock)
        .language_model("anthropic/claude-haiku-4.5")
        .stream(text_request("hi"))
        .await
        .expect("stream establishes")
        .collect_result()
        .await
        .expect("stream completes");

    assert_eq!(
        result.provider_metadata.get("openrouter"),
        Some(&json!({
            "provider": "Amazon Bedrock",
            "usage": {
                "cost": usage["cost"],
                "is_byok": usage["is_byok"],
                "cost_details": usage["cost_details"]
            }
        }))
    );
}

#[tokio::test]
async fn keepalive_comments_and_streamed_details() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        ": OPENROUTER PROCESSING",
        r#"{"id":"gen-4","object":"chat.completion.chunk","created":1,"model":"anthropic/claude-sonnet-5","choices":[{"index":0,"delta":{"role":"assistant","reasoning":"think ","reasoning_details":[{"type":"reasoning.text","text":"think ","index":0,"format":"anthropic-claude-v1"}]},"finish_reason":null}]}"#,
        ": OPENROUTER PROCESSING",
        r#"{"id":"gen-4","object":"chat.completion.chunk","created":1,"model":"anthropic/claude-sonnet-5","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}"#,
        r#"{"id":"gen-4","object":"chat.completion.chunk","created":1,"model":"anthropic/claude-sonnet-5","choices":[{"index":0,"delta":{},"finish_reason":"stop","native_finish_reason":"end_turn"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost":0.0003}}"#,
        "[DONE]",
    ]);
    let provider = openrouter(&mock);
    let events = drain(
        provider
            .language_model("anthropic/claude-sonnet-5")
            .stream(text_request("hi"))
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
    assert!(reasoning.provider_metadata.get("openrouter").is_some());
    let Some(StreamEvent::Finish { finish, usage }) = events.last() else {
        panic!()
    };
    assert_eq!(finish.raw.as_deref(), Some("end_turn"));
    assert_eq!(usage.input_tokens, Some(10));
}

#[tokio::test]
async fn midstream_error_chunk_fails_stream() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"gen-5","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"par"},"finish_reason":null}]}"#,
        r#"{"id":"gen-5","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":""},"finish_reason":"error"}],"error":{"code":429,"message":"Rate limit exceeded","metadata":{"error_type":"rate_limit_exceeded"}}}"#,
    ]);
    let provider = openrouter(&mock);
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
        .expect("stream error");
    assert_eq!(error.kind(), ErrorKind::RateLimited);
    let Some(StreamEvent::Finish { finish, .. }) = events.last() else {
        panic!()
    };
    assert_eq!(finish.reason, FinishReason::Error);
}

#[tokio::test]
async fn midstream_content_policy_error_is_typed() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"gen-policy","model":"openai/gpt-5.6-sol","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}"#,
        r#"{"id":"gen-policy","model":"openai/gpt-5.6-sol","choices":[],"error":{"code":403,"message":"Request blocked by provider guardrail","metadata":{"error_type":"content_policy_violation"}}}"#,
    ]);

    let events = drain(
        openrouter(&mock)
            .language_model("openai/gpt-5.6-sol")
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
async fn invalid_utf8_is_replaced_without_dropping_prior_frames() {
    let mock = MockTransport::shared();
    let mut chunk = br#"data: {"id":"gen-invalid-utf8","model":"m","choices":[{"index":0,"delta":{"content":"prefix"},"finish_reason":null}]}

data: {"id":"gen-invalid-utf8","choices":[{"index":0,"delta":{"content":""#
        .to_vec();
    chunk.push(0xFF);
    chunk.extend_from_slice(
        br#""},"finish_reason":null}]}

data: {"id":"gen-invalid-utf8","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

"#,
    );
    mock.push_stream_chunks(
        200,
        headers(&[("content-type", "text/event-stream")]),
        vec![bytes::Bytes::from(chunk)],
    );

    let events = drain(
        openrouter(&mock)
            .language_model("m")
            .stream(text_request("hi"))
            .await
            .expect("stream establishes"),
    )
    .await;

    let text = events.iter().filter_map(|event| match event {
        StreamEvent::TextDelta { delta, .. } => Some(delta.as_str()),
        _ => None,
    });
    assert_eq!(text.collect::<String>(), "prefix�");
    assert_terminal_contract(&events);
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::Error { .. }))
    );
}

#[tokio::test]
async fn visible_reasoning_details_emit_deltas_without_duplicating_plaintext() {
    for include_plaintext in [false, true] {
        let mock = MockTransport::shared();
        let mut frames = Vec::new();
        for detail in [
            json!({"type": "reasoning.summary", "summary": "First, ", "index": 0}),
            json!({"type": "reasoning.summary", "summary": "check. ", "index": 0}),
            json!({"type": "reasoning.text", "text": "Then answer.", "index": 1}),
            json!({"type": "reasoning.encrypted", "data": "OPAQUE", "index": 2}),
        ] {
            let mut delta = json!({"reasoning_details": [detail.clone()]});
            if include_plaintext {
                delta["reasoning"] = detail
                    .get("text")
                    .or_else(|| detail.get("summary"))
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
            }
            frames.push(json!({"choices": [{"index": 0, "delta": delta}]}).to_string());
        }
        frames.push(
            json!({"choices": [{"index": 0, "delta": {"content": "Done"},
            "finish_reason": "stop"}]})
            .to_string(),
        );
        frames.push("[DONE]".into());
        mock.push_sse(&frames.iter().map(String::as_str).collect::<Vec<_>>());
        let events = drain(
            openrouter(&mock)
                .language_model("model")
                .stream(text_request("hi"))
                .await
                .unwrap(),
        )
        .await;
        assert_terminal_contract(&events);
        let visible: String = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ReasoningDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            visible, "First, check. Then answer.",
            "include_plaintext={include_plaintext}"
        );
        let part = events
            .iter()
            .find_map(|event| match event {
                StreamEvent::ReasoningEnd { part, .. } => Some(part),
                _ => None,
            })
            .unwrap();
        assert_eq!(part.visible_text(), visible);
        assert_eq!(
            part.provider_metadata.get("openrouter").unwrap()["reasoning_details"][2]["data"],
            "OPAQUE"
        );
    }
}
