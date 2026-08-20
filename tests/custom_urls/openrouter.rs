use caido_ai::transport::mock::MockTransport;
use caido_ai::{Credentials, Message, ProviderConfig, Request};
use serde_json::json;

use crate::common::provider_with;

#[tokio::test]
async fn provider_default_headers_and_request_extra_headers() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({"id": "x", "object": "chat.completion", "created": 1, "model": "m",
               "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok",
                            "refusal": null}, "finish_reason": "stop", "logprobs": null}]}),
    );
    let provider = provider_with(
        &mock,
        ProviderConfig::openrouter(Credentials::api_key("sk-or"))
            .with_header("HTTP-Referer", "https://caido.io")
            .with_header("X-OpenRouter-Title", "Caido"),
    );
    let request = Request::builder()
        .message(Message::user("hi"))
        .extra_header("x-trace-id", "trace-1")
        .build();
    provider
        .language_model("openai/gpt-5.6")
        .generate(request)
        .await
        .unwrap();
    let http = &mock.requests()[0];
    for (expected_name, expected_value) in [
        ("HTTP-Referer", "https://caido.io"),
        ("X-OpenRouter-Title", "Caido"),
        ("x-trace-id", "trace-1"),
    ] {
        assert!(
            http.headers.iter().any(|(name, value)| {
                name.eq_ignore_ascii_case(expected_name) && value == expected_value
            }),
            "missing header {expected_name}"
        );
    }
}
