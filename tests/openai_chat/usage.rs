use super::*;

#[tokio::test]
async fn output_usage_does_not_double_count_included_reasoning_tokens() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "openai-usage", "model": "gpt-5.6",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 215,
                "completion_tokens": 575,
                "total_tokens": 790,
                "completion_tokens_details": {"reasoning_tokens": 572}
            }
        }),
    );
    let result = openai_chat(&mock)
        .language_model("gpt-5.6")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");

    assert_eq!(result.usage.output_tokens, Some(575));
}

#[tokio::test]
async fn usage_only_chunk_with_empty_choices_is_tolerated() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"id":"","object":"","created":0,"model":"","choices":[],"prompt_filter_results":[{"prompt_index":0}]}"#,
        r#"{"id":"c6","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}"#,
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
    assert_eq!(result.text(), "ok");
}
