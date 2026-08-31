use super::*;

#[tokio::test]
async fn compaction_streams_and_pauses() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"message_start","message":{"id":"msg_6","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":150000,"output_tokens":1}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"compaction"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"Summary of the long conversation."}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"compaction","stop_sequence":null},"usage":{"output_tokens":40}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let provider = anthropic(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .compaction(Compaction {
            pause_after_compaction: Some(true),
            ..Compaction::enabled()
        })
        .build();
    let result = provider
        .language_model("claude-opus-5")
        .stream(request)
        .await
        .unwrap()
        .collect_result()
        .await
        .expect("collect");

    assert_eq!(result.finish.reason, FinishReason::Paused);
    assert_eq!(result.finish.raw.as_deref(), Some("compaction"));
    let compaction = result
        .content
        .iter()
        .find_map(|part| match part {
            AssistantPart::Compaction(part) => Some(part.clone()),
            _ => None,
        })
        .expect("compaction part");
    assert_eq!(
        compaction.content.as_deref(),
        Some("Summary of the long conversation.")
    );

    let mock2 = MockTransport::shared();
    mock2.push_json(200, &minimal_message());
    let provider2 = anthropic(&mock2);
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(result.to_assistant_message())
        .message(Message::user("go on"))
        .build();
    provider2
        .language_model("claude-opus-5")
        .generate(request)
        .await
        .expect("generate succeeds");
    let body = mock2.request_json(0);
    let messages = body["messages"].as_array().unwrap();
    assert!(messages.iter().any(|message| {
        message["content"]
            .as_array()
            .is_some_and(|blocks| blocks.iter().any(|block| block["type"] == "compaction"))
    }));
}

#[tokio::test]
async fn replaying_a_compaction_block_enables_the_beta() {
    let mock = MockTransport::shared();
    mock.push_json(200, &minimal_message());
    let provider = anthropic(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(Message::Assistant {
            content: vec![AssistantPart::Compaction(llmwire::CompactionPart {
                id: None,
                content: Some("Earlier we discussed X.".into()),
                encrypted_content: None,
            })],
            provider_metadata: ProviderMetadata::default(),
        })
        .message(Message::user("continue"))
        .build();
    provider
        .language_model("claude-opus-5")
        .generate(request)
        .await
        .expect("generate succeeds");
    let requests = mock.requests();
    let beta = header(&requests[0], "anthropic-beta")
        .expect("beta header required to send compaction blocks");
    assert!(beta.contains("compact-2026-01-12"), "{beta}");
    let body: serde_json::Value =
        serde_json::from_slice(mock.requests()[0].body.as_ref().unwrap()).unwrap();
    assert_eq!(
        body["context_management"]["edits"][0]["type"],
        serde_json::json!("compact_20260112"),
        "history with compaction blocks must re-send the compact strategy"
    );
}

#[tokio::test]
async fn paused_compaction_usage_is_summed_from_iterations() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "msg_c", "type": "message", "role": "assistant", "model": "claude-opus-5",
            "content": [{"type": "compaction", "content": "Earlier we discussed tulips."}],
            "stop_reason": "compaction", "stop_sequence": null,
            "usage": {
                "input_tokens": 0, "output_tokens": 0,
                "iterations": [
                    {"type": "compaction", "input_tokens": 69_313, "output_tokens": 414,
                     "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
                    {"type": "message", "input_tokens": 1200, "output_tokens": 7,
                     "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
                ]
            }
        }),
    );
    let provider = anthropic(&mock);
    let result = provider
        .language_model("claude-opus-5")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");
    assert_eq!(result.finish.reason, FinishReason::Paused);
    assert_eq!(result.usage.input_tokens, Some(70_513));
    assert_eq!(result.usage.output_tokens, Some(421));
    let compactions: Vec<_> = result.compactions().collect();
    assert_eq!(compactions.len(), 1);
    assert_eq!(
        compactions[0].content.as_deref(),
        Some("Earlier we discussed tulips.")
    );
}

#[tokio::test]
async fn foreign_opaque_compaction_is_rejected() {
    let mock = MockTransport::shared();
    let provider = anthropic(&mock);
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(Message::Assistant {
            content: vec![llmwire::AssistantPart::Compaction(
                llmwire::CompactionPart {
                    id: Some("cmp_1".into()),
                    content: None,
                    encrypted_content: Some("OPAQUE".into()),
                },
            )],
            provider_metadata: llmwire::ProviderMetadata::default(),
        })
        .message(Message::user("next"))
        .build();
    let error = provider
        .language_model("claude-haiku-4-5")
        .generate(request)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), llmwire::ErrorKind::UnsupportedContent);
    assert!(mock.requests().is_empty());
}
