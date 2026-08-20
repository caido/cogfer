use super::*;

#[tokio::test]
async fn blocking_response_preserves_cost_and_routing_metadata() {
    let mock = MockTransport::shared();
    let usage = json!({
        "prompt_tokens": 25,
        "completion_tokens": 7,
        "total_tokens": 32,
        "cost": 0.0000067,
        "is_byok": false,
        "cost_details": {
            "upstream_inference_cost": 0.0000067,
            "upstream_inference_prompt_cost": 0.0000025,
            "upstream_inference_completions_cost": 0.0000042
        }
    });
    mock.push_json(
        200,
        &json!({
            "id": "gen-metadata", "model": "openai/gpt-5.6-luna", "provider": "OpenAI",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": usage
        }),
    );
    let result = openrouter(&mock)
        .language_model("openai/gpt-5.6-luna")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");

    assert_eq!(
        result.provider_metadata.get("openrouter"),
        Some(&json!({
            "provider": "OpenAI",
            "usage": {
                "cost": usage["cost"],
                "is_byok": usage["is_byok"],
                "cost_details": usage["cost_details"]
            }
        }))
    );
}

#[tokio::test]
async fn output_usage_does_not_double_count_reasoning_tokens() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "gen-usage", "model": "anthropic/claude-haiku-4.5",
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
    let result = openrouter(&mock)
        .language_model("anthropic/claude-haiku-4.5")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");

    assert_eq!(result.usage.output_tokens, Some(575));
}

#[tokio::test]
async fn error_inside_http_200_body() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({"error": {"code": 402, "message": "Insufficient credits",
                 "metadata": {"error_type": "payment_required"}}}),
    );
    let provider = openrouter(&mock);
    let error = provider
        .language_model("m")
        .generate(text_request("hi"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Permission);
    assert_eq!(error.status(), Some(402));
}

#[tokio::test]
async fn content_policy_error_inside_http_200_body_is_typed() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({"error": {
            "code": 403,
            "message": "Request blocked by provider guardrail",
            "metadata": {"error_type": "content_policy_violation"}
        }}),
    );

    let error = openrouter(&mock)
        .language_model("openai/gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect_err("policy failure is an error");

    assert_eq!(error.kind(), ErrorKind::ContentPolicy);
    assert_eq!(error.code(), Some("content_policy_violation"));
    assert_eq!(error.status(), Some(403));
}

#[tokio::test]
async fn error_inside_first_choice_is_not_returned_as_partial_success() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({
            "id": "gen-error",
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "partial output"},
                "finish_reason": "error",
                "error": {
                    "code": 502,
                    "message": "Provider disconnected mid-generation",
                    "metadata": {"error_type": "provider_unavailable"}
                }
            }]
        }),
    );

    let error = openrouter(&mock)
        .language_model("m")
        .generate(text_request("hi"))
        .await
        .expect_err("choice-level provider error must fail generation");

    assert_eq!(error.kind(), ErrorKind::Overloaded);
    assert_eq!(error.status(), Some(502));
    assert_eq!(error.code(), Some("provider_unavailable"));
}
