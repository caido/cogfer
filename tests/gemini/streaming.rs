use super::*;

#[tokio::test]
async fn streaming_thoughts_text_and_function_calls() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Thinking about it...","thought":true}]},"index":0}],"modelVersion":"gemini-2.5-pro","responseId":"r3"}"#,
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Let me call the tool."}]},"index":0}],"responseId":"r3"}"#,
        r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"location":"Paris"}},"thoughtSignature":"SIG_S"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":42,"candidatesTokenCount":20,"thoughtsTokenCount":100,"totalTokenCount":162},"responseId":"r3"}"#,
    ]);
    let provider = gemini(&mock);
    let events = drain(
        provider
            .language_model("gemini-2.5-pro")
            .stream(tool_request("weather"))
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
            "reasoning-start",
            "reasoning-delta",
            "reasoning-end",
            "text-start",
            "text-delta",
            "text-end",
            "tool-input-start",
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
        .expect("tool call");
    assert_eq!(
        call.provider_metadata
            .get("gemini")
            .and_then(|namespace| namespace.get("thought_signature"))
            .and_then(serde_json::Value::as_str),
        Some("SIG_S")
    );
    let Some(StreamEvent::Finish { finish, usage }) = events.last() else {
        panic!()
    };
    assert_eq!(finish.reason, FinishReason::ToolCalls);
    assert_eq!(usage.output_tokens, Some(120));
}

#[tokio::test]
async fn response_metadata_emits_fields_that_arrive_in_later_chunks() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"responseId":"response-1","candidates":[]}"#,
        r#"{"modelVersion":"gemini-3-flash","candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP","index":0}]}"#,
    ]);

    let events = drain(
        gemini(&mock)
            .language_model("gemini-3-flash")
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
    assert_eq!(metadata[0].id.as_deref(), Some("response-1"));
    assert_eq!(metadata[0].model, None);
    assert_eq!(metadata[1].id, None);
    assert_eq!(metadata[1].model.as_deref(), Some("gemini-3-flash"));
}

#[tokio::test]
async fn streamed_safety_block_preserves_prompt_feedback() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"promptFeedback":{"blockReason":"SAFETY","safetyRatings":[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","probability":"HIGH","blocked":true}]},"usageMetadata":{"promptTokenCount":12,"totalTokenCount":12},"responseId":"blocked-1"}"#,
    ]);

    let events = drain(
        gemini(&mock)
            .language_model("gemini-3.6-flash")
            .stream(text_request("hi"))
            .await
            .expect("stream establishes"),
    )
    .await;

    assert_terminal_contract(&events);
    let metadata = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ProviderMetadata(metadata) => metadata.get("gemini"),
            _ => None,
        })
        .expect("prompt feedback metadata");
    assert_eq!(metadata["promptFeedback"]["blockReason"], json!("SAFETY"));
    assert_eq!(
        metadata["promptFeedback"]["safetyRatings"][0]["blocked"],
        json!(true)
    );
    let Some(StreamEvent::Finish { finish, .. }) = events.last() else {
        panic!("missing finish");
    };
    assert_eq!(finish.reason, FinishReason::ContentFilter);
}

#[tokio::test]
async fn streamed_candidate_filter_preserves_safety_details() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"candidates":[{"content":{"parts":[]},"finishReason":"SAFETY","finishMessage":"The response was blocked by a safety filter.","safetyRatings":[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","probability":"HIGH","blocked":true}]}],"modelVersion":"gemini-3.6-flash","responseId":"filtered-2"}"#,
    ]);

    let events = drain(
        gemini(&mock)
            .language_model("gemini-3.6-flash")
            .stream(text_request("hi"))
            .await
            .expect("stream establishes"),
    )
    .await;

    assert_terminal_contract(&events);
    let metadata = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ProviderMetadata(metadata) => metadata.get("gemini"),
            _ => None,
        })
        .expect("candidate safety metadata");
    assert_eq!(
        metadata["finishMessage"],
        "The response was blocked by a safety filter."
    );
    assert_eq!(metadata["safetyRatings"][0]["blocked"], true);
    let Some(StreamEvent::Finish { finish, .. }) = events.last() else {
        panic!("missing finish");
    };
    assert_eq!(finish.reason, FinishReason::ContentFilter);
}

#[tokio::test]
async fn empty_text_part_with_signature_is_preserved() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Answer."}]},"index":0}],"responseId":"r4"}"#,
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"","thoughtSignature":"TAIL_SIG"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7},"responseId":"r4"}"#,
    ]);
    let provider = gemini(&mock);
    let result = provider
        .language_model("gemini-3-flash")
        .stream(text_request("hi"))
        .await
        .unwrap()
        .collect_result()
        .await
        .expect("collect");
    assert_eq!(result.text(), "Answer.");
    let AssistantPart::Text {
        provider_metadata, ..
    } = &result.content[0]
    else {
        panic!("expected text part");
    };
    assert_eq!(
        provider_metadata
            .get("gemini")
            .and_then(|namespace| namespace.get("thought_signature"))
            .and_then(serde_json::Value::as_str),
        Some("TAIL_SIG")
    );
}

#[tokio::test]
async fn malformed_function_call_is_error_finish() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"MALFORMED_FUNCTION_CALL","index":0}],"usageMetadata":{"promptTokenCount":5,"totalTokenCount":5},"responseId":"r5"}"#,
    ]);
    let provider = gemini(&mock);
    let events = drain(
        provider
            .language_model("gemini-2.5-flash")
            .stream(tool_request("x"))
            .await
            .unwrap(),
    )
    .await;
    assert_terminal_contract(&events);
    let Some(StreamEvent::Finish { finish, .. }) = events.last() else {
        panic!()
    };
    assert_eq!(finish.reason, FinishReason::Error);
    assert_eq!(finish.raw.as_deref(), Some("MALFORMED_FUNCTION_CALL"));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::Error { .. })),
        "expected an error event: {:?}",
        kinds(&events)
    );
}

#[tokio::test]
async fn max_tokens_does_not_promote_a_function_call() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"delete","args":{"path":"/tmp"}}}]},"finishReason":"MAX_TOKENS","index":0}],"responseId":"r_partial"}"#,
    ]);
    let events = drain(
        gemini(&mock)
            .language_model("gemini-2.5-flash")
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
async fn signature_only_part_is_preserved() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Answer."}]},"index":0}],"responseId":"r5"}"#,
        r#"{"candidates":[{"content":{"role":"model","parts":[{"thoughtSignature":"ONLY_SIG"}]},"finishReason":"STOP","index":0}],"responseId":"r5"}"#,
    ]);
    let provider = gemini(&mock);
    let result = provider
        .language_model("gemini-3-flash")
        .stream(text_request("hi"))
        .await
        .unwrap()
        .collect_result()
        .await
        .expect("collect");
    let signature = result.content.iter().find_map(|part| match part {
        AssistantPart::Text {
            provider_metadata, ..
        } => provider_metadata
            .get("gemini")
            .and_then(|namespace| namespace.get("thought_signature"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
        _ => None,
    });
    assert_eq!(signature.as_deref(), Some("ONLY_SIG"));

    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "candidates": [{"content": {"role": "model",
                "parts": [{"text": "Answer."}, {"thoughtSignature": "ONLY_SIG"}]},
                "finishReason": "STOP", "index": 0}],
            "responseId": "r6"
        }),
    );
    let provider = gemini(&mock);
    let result = provider
        .language_model("gemini-3-flash")
        .generate(text_request("hi"))
        .await
        .expect("generate");
    let signature = result.content.iter().find_map(|part| match part {
        AssistantPart::Text {
            provider_metadata, ..
        } => provider_metadata
            .get("gemini")
            .and_then(|namespace| namespace.get("thought_signature"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
        _ => None,
    });
    assert_eq!(signature.as_deref(), Some("ONLY_SIG"));
}

#[tokio::test]
async fn grounding_metadata_streams_citations_and_provider_metadata() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Paris is the capital."}]},"index":0}],"modelVersion":"gemini-2.5-pro","responseId":"rg"}"#,
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" It is in France."}]},"finishReason":"STOP","index":0,"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://example.com/paris","title":"Paris"}}],"webSearchQueries":["capital of france"]}}],"usageMetadata":{"promptTokenCount":8,"candidatesTokenCount":9,"totalTokenCount":17},"responseId":"rg"}"#,
    ]);
    let provider = gemini(&mock);
    let events = drain(
        provider
            .language_model("gemini-2.5-pro")
            .stream(text_request("capital of France?"))
            .await
            .unwrap(),
    )
    .await;
    assert_terminal_contract(&events);

    let citation = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::Citation { text_id, citation } => {
                assert!(text_id.is_none(), "gemini grounding is response-level");
                Some(citation.clone())
            }
            _ => None,
        })
        .expect("citation event streams");
    assert_eq!(citation.url.as_deref(), Some("https://example.com/paris"));
    assert_eq!(citation.title.as_deref(), Some("Paris"));

    let grounding = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ProviderMetadata(metadata) => metadata.get("gemini").cloned(),
            _ => None,
        })
        .expect("grounding metadata surfaces");
    assert_eq!(
        grounding["groundingMetadata"]["webSearchQueries"][0],
        "capital of france"
    );
}

#[tokio::test]
async fn midstream_error_frames_report_the_envelope_status() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]},"index":0}]}"#,
        r#"{"error":{"code":503,"message":"The model is overloaded.","status":"UNAVAILABLE"}}"#,
    ]);
    let events = drain(
        gemini(&mock)
            .language_model("gemini-2.5-flash")
            .stream(text_request("x"))
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
    assert_eq!(error.status(), Some(503));
    assert_eq!(error.code(), Some("UNAVAILABLE"));
    assert!(error.retryable());
}
