use super::*;

#[tokio::test]
async fn request_golden() {
    let mock = MockTransport::shared();
    mock.push_sse(&completed_transcript());
    let provider = chatgpt(&mock);
    let model = provider.language_model("gpt-5.6-sol");

    let request = Request::builder()
        .system("Be helpful.")
        .message(Message::user("hi"))
        .message(Message::assistant("hello"))
        .system("Mid-conversation instruction.")
        .message(Message::user("bye"))
        .temperature(0.2)
        .max_output_tokens(500)
        .build();

    let result = model.generate(request).await.expect("generate succeeds");
    assert_eq!(result.text(), "ok");
    assert_eq!(result.finish.reason, FinishReason::Stop);
    assert_eq!(result.usage.input_tokens, Some(12));

    let http = &mock.requests()[0];
    assert_eq!(
        http.url.as_str(),
        "https://chatgpt.com/backend-api/codex/responses"
    );
    assert_eq!(
        header(http, "authorization"),
        Some("Bearer chatgpt-access-token")
    );
    assert_eq!(header(http, "accept"), Some("text/event-stream"));
    assert_eq!(header(http, "originator"), Some("caido-ai"));
    let session_id = header(http, "session_id").expect("session_id header");
    assert!(
        uuid::Uuid::parse_str(session_id).is_ok(),
        "session_id should be a UUID, got {session_id:?}"
    );

    let body = mock.request_json(0);
    assert_eq!(
        body["instructions"],
        json!("Be helpful.\n\nMid-conversation instruction.")
    );
    let roles: Vec<&str> = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|item| item["role"].as_str())
        .collect();
    assert_eq!(roles, vec!["user", "assistant", "user"]);
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(body["stream"], json!(true));
    assert!(body.get("stream_options").is_none());
    assert!(body.get("temperature").is_none());
    assert!(body.get("max_output_tokens").is_none());

    let warned: Vec<&str> = result
        .warnings
        .iter()
        .map(|warning| warning.message.as_str())
        .collect();
    assert!(
        warned.iter().any(|message| message.contains("temperature")),
        "expected a temperature warning, got {warned:?}"
    );
    assert!(
        warned
            .iter()
            .any(|message| message.contains("max_output_tokens")),
        "expected a max_output_tokens warning, got {warned:?}"
    );
}

#[tokio::test]
async fn default_instructions_fill_in_when_no_system_content() {
    let mock = MockTransport::shared();
    mock.push_sse(&completed_transcript());
    let provider = chatgpt(&mock);

    provider
        .language_model("gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");

    assert_eq!(
        mock.request_json(0)["instructions"],
        json!("You are ChatGPT, a helpful AI assistant.")
    );
}

#[tokio::test]
async fn caller_system_content_passes_through_verbatim() {
    let mock = MockTransport::shared();
    mock.push_sse(&completed_transcript());
    let provider = chatgpt(&mock);

    provider
        .language_model("gpt-5.6-sol")
        .generate(
            Request::builder()
                .system("Answer in French.")
                .message(Message::user("hi"))
                .build(),
        )
        .await
        .expect("generate succeeds");

    assert_eq!(
        mock.request_json(0)["instructions"],
        json!("Answer in French.")
    );
}

#[tokio::test]
async fn streaming_normalizes_the_transcript() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_s1","model":"gpt-5.6-sol","status":"in_progress"}}"#,
        r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress","content":[]}}"#,
        r#"{"type":"response.content_part.added","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}"#,
        r#"{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"bonjour"}"#,
        r#"{"type":"response.content_part.done","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"bonjour","annotations":[]}}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"bonjour","annotations":[]}]}}"#,
        r#"{"type":"response.completed","response":{"id":"resp_s1","status":"completed","output":[],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}"#,
    ]);
    let provider = chatgpt(&mock);

    let events = drain(
        provider
            .language_model("gpt-5.6-sol")
            .stream(text_request("hi"))
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
            "text-start",
            "text-delta",
            "text-end",
            "finish",
        ]
    );
}

#[tokio::test]
async fn generate_reconstructs_items_when_terminal_output_is_empty() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_r1","model":"gpt-5.6-sol","status":"in_progress"}}"#,
        r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress","content":[]}}"#,
        r#"{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"Checking the weather."}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Checking the weather.","annotations":[]}]}}"#,
        r#"{"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"get_weather","arguments":"","status":"in_progress"}}"#,
        r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":1,"delta":"{\"city\":\"Paris\"}"}"#,
        r#"{"type":"response.output_item.done","output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}","status":"completed"}}"#,
        r#"{"type":"response.completed","response":{"id":"resp_r1","status":"completed","model":"gpt-5.6-sol","output":[],"usage":{"input_tokens":7,"output_tokens":9,"total_tokens":16}}}"#,
        "[DONE]",
    ]);
    let provider = chatgpt(&mock);

    let result = provider
        .language_model("gpt-5.6-sol")
        .generate(text_request("weather?"))
        .await
        .expect("generate succeeds via transcript reconstruction");

    assert_eq!(result.text(), "Checking the weather.");
    let call = result.tool_calls().next().expect("tool call survives");
    assert_eq!(call.name, "get_weather");
    assert_eq!(call.arguments, r#"{"city":"Paris"}"#);
    assert_eq!(result.finish.reason, FinishReason::ToolCalls);
    assert_eq!(result.usage.input_tokens, Some(7));
    assert_eq!(
        result.provider_metadata.get("caido-ai").unwrap()["profile"],
        "chatgpt"
    );
}

#[tokio::test]
async fn nested_error_envelopes_keep_their_details() {
    let failed_transcript = [
        r#"{"type":"response.created","response":{"id":"resp_f1","model":"gpt-5.6-sol","status":"in_progress"}}"#,
        r#"{"type":"error","error":{"type":"server_error","code":"server_error","message":"An error occurred while processing your request. Include request ID b5c5c4dc."},"sequence_number":2}"#,
        r#"{"type":"response.failed","response":{"id":"resp_f1","status":"failed","output":[],"error":{"code":"server_error","message":"An error occurred while processing your request. Include request ID b5c5c4dc."}}}"#,
    ];

    let mock = MockTransport::shared();
    mock.push_sse(&failed_transcript);
    let provider = chatgpt(&mock);
    let error = provider
        .language_model("gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect_err("failed transcript should fail generate");
    assert!(error.message().contains("request ID b5c5c4dc"), "{error}");
    assert_eq!(error.code(), Some("server_error"));
    assert_eq!(error.origin(), Some("chatgpt"));

    let mock = MockTransport::shared();
    mock.push_sse(&failed_transcript);
    let provider = chatgpt(&mock);
    let events = drain(
        provider
            .language_model("gpt-5.6-sol")
            .stream(text_request("hi"))
            .await
            .expect("stream establishes"),
    )
    .await;
    let stream_error = events
        .iter()
        .find_map(|event| match event {
            caido_ai::StreamEvent::Error { error } => Some(error),
            _ => None,
        })
        .expect("stream reports the error event");
    assert!(
        stream_error.message().contains("request ID b5c5c4dc"),
        "{stream_error}"
    );
    assert_eq!(stream_error.code(), Some("server_error"));
    assert_eq!(stream_error.origin(), Some("chatgpt"));
}

#[tokio::test]
async fn generate_surfaces_transcript_errors() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_e1","status":"in_progress"}}"#,
        r#"{"type":"error","code":"rate_limit_exceeded","message":"slow down"}"#,
    ]);
    let provider = chatgpt(&mock);

    let error = provider
        .language_model("gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect_err("transcript error should fail generate");
    assert!(error.message().contains("slow down"), "{error}");
    assert_eq!(error.code(), Some("rate_limit_exceeded"));
}

#[tokio::test]
async fn generate_without_terminal_event_is_malformed() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_t1","status":"in_progress"}}"#,
        r#"{"type":"response.output_text.delta","item_id":"msg_1","delta":"partial"}"#,
    ]);
    let provider = chatgpt(&mock);

    let error = provider
        .language_model("gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect_err("truncated transcript should fail");
    assert_eq!(error.kind(), ErrorKind::MalformedResponse);
}

#[tokio::test]
async fn account_id_header_via_provider_config() {
    let mock = MockTransport::shared();
    mock.push_sse(&completed_transcript());
    let provider = provider_with(
        &mock,
        ProviderConfig::chatgpt(Credentials::bearer("tok")).with_header(
            HeaderName::from_static("chatgpt-account-id"),
            HeaderValue::from_static("acct_static"),
        ),
    );

    provider
        .language_model("gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");

    let http = &mock.requests()[0];
    assert!(
        http.headers
            .iter()
            .any(|(name, value)| name == "chatgpt-account-id" && value == "acct_static")
    );
}

#[tokio::test]
async fn generate_classifies_transcript_errors_like_streaming() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_e2","status":"in_progress"}}"#,
        r#"{"type":"error","error":{"code":"content_policy_violation","message":"blocked"}}"#,
    ]);
    let error = chatgpt(&mock)
        .language_model("gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect_err("transcript error should fail generate");

    assert_eq!(error.kind(), ErrorKind::ContentPolicy);
    assert_eq!(error.code(), Some("content_policy_violation"));
    assert_eq!(error.origin(), Some("chatgpt"));
}

#[tokio::test]
async fn chatgpt_turns_do_not_leak_opaque_state_to_other_responses_backends() {
    // A ChatGPT turn with encrypted reasoning, as recorded into history.
    let chatgpt_mock = MockTransport::shared();
    chatgpt_mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_x1","model":"gpt-5.6-sol","status":"in_progress"}}"#,
        r#"{"type":"response.completed","response":{"id":"resp_x1","status":"completed","model":"gpt-5.6-sol","output":[{"id":"rs_x1","type":"reasoning","summary":[],"encrypted_content":"ENC-CHATGPT"},{"id":"msg_x1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}"#,
        "[DONE]",
    ]);
    let result = chatgpt(&chatgpt_mock)
        .language_model("gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect("chatgpt generate succeeds");
    assert_eq!(
        result.provider_metadata.get("caido-ai").unwrap()["profile"],
        "chatgpt"
    );

    // Switching models: the same history replayed to the OpenAI API must not
    // carry ChatGPT-encrypted items the API cannot decrypt.
    let openai_mock = MockTransport::shared();
    openai_mock.push_json(
        200,
        &json!({
            "id": "resp_x2", "object": "response", "status": "completed", "model": "gpt-5.6-luna",
            "output": [{"id": "msg_x2", "type": "message", "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
            "usage": {"input_tokens": 9, "output_tokens": 1, "total_tokens": 10}
        }),
    );
    let follow_up = Request::builder()
        .message(Message::user("hi"))
        .message(result.to_assistant_message())
        .message(Message::user("continue"))
        .build();
    let replayed = openai_responses(&openai_mock)
        .language_model("gpt-5.6-luna")
        .generate(follow_up)
        .await
        .expect("openai generate succeeds");

    let body = openai_mock.request_json(0);
    let items = body["input"].as_array().unwrap();
    assert!(
        items.iter().all(|item| item["type"] != json!("reasoning")),
        "chatgpt reasoning must not reach the openai api: {items:?}"
    );
    assert!(!body.to_string().contains("ENC-CHATGPT"));
    assert!(
        replayed
            .warnings
            .iter()
            .any(|warning| warning.message.contains("chatgpt")),
        "{:?}",
        replayed.warnings
    );
}
