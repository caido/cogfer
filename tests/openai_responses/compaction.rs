use super::*;

#[tokio::test]
async fn compaction_item_streams_and_replays() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_9","model":"gpt-5.6"},"sequence_number":0}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"cmp_1","type":"compaction","encrypted_content":"CMP_BLOB"},"sequence_number":1}"#,
        r#"{"type":"response.output_item.added","output_index":1,"item":{"id":"msg_9","type":"message","role":"assistant","status":"in_progress","content":[]},"sequence_number":2}"#,
        r#"{"type":"response.output_text.delta","item_id":"msg_9","output_index":1,"content_index":0,"delta":"done","sequence_number":3}"#,
        r#"{"type":"response.completed","response":{"id":"resp_9","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}},"sequence_number":4}"#,
    ]);
    let provider = openai_responses(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .compaction(Compaction::enabled())
        .build();
    let stream = provider
        .language_model("gpt-5.6")
        .stream(request)
        .await
        .unwrap();
    let result = stream.collect_result().await.expect("collect");

    let compaction = result
        .content
        .iter()
        .find_map(|part| match part {
            AssistantPart::Compaction(part) => Some(part.clone()),
            _ => None,
        })
        .expect("compaction part in result");
    assert_eq!(compaction.encrypted_content.as_deref(), Some("CMP_BLOB"));
    assert_eq!(compaction.id.as_deref(), Some("cmp_1"));

    let mock2 = MockTransport::shared();
    mock2.push_json(200, &minimal_completed());
    let provider2 = openai_responses(&mock2);
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(result.to_assistant_message())
        .message(Message::user("continue"))
        .build();
    provider2
        .language_model("gpt-5.6")
        .generate(request)
        .await
        .expect("generate succeeds");
    let body = mock2.request_json(0);
    let input = body["input"].as_array().unwrap();
    assert!(input.iter().any(|item| item["type"] == "compaction"
        && item["encrypted_content"] == "CMP_BLOB"
        && item["id"] == "cmp_1"));
}

#[tokio::test]
async fn summary_compaction_is_flattened_to_assistant_text() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_completed());
    let provider = openai_responses(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(Message::Assistant {
            content: vec![AssistantPart::Compaction(CompactionPart {
                id: None,
                content: Some("Earlier: the user asked about Paris.".into()),
                encrypted_content: None,
            })],
            provider_metadata: ProviderMetadata::default(),
        })
        .message(Message::user("next"))
        .build();

    let result = provider
        .language_model("gpt-5.6")
        .generate(request)
        .await
        .expect("generate succeeds");

    let items = mock.request_json(0)["input"].clone();
    let items = items.as_array().unwrap();
    assert_eq!(items[1]["role"], "assistant");
    assert_eq!(
        items[1]["content"][0]["text"],
        "Earlier: the user asked about Paris."
    );
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.message.contains("flattened")),
        "{:?}",
        result.warnings
    );
}
