use caido_ai::transport::HeaderName;
use caido_ai::transport::mock::MockTransport;
use caido_ai::{Credentials, ProviderConfig};
use serde_json::json;
use url::Url;

use crate::common::{header, provider_with, text_request};

fn chat_ok() -> serde_json::Value {
    json!({"id": "x", "object": "chat.completion", "created": 1, "model": "m",
           "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok",
                        "refusal": null}, "finish_reason": "stop", "logprobs": null}]})
}

#[tokio::test]
async fn litellm_prefix_and_trailing_slash_are_preserved() {
    for base in [
        "http://localhost:4000/litellm/v1",
        "http://localhost:4000/litellm/v1/",
    ] {
        let mock = MockTransport::shared();
        mock.push_json(200, &chat_ok());
        let provider = provider_with(
            &mock,
            ProviderConfig::openai_chat(Credentials::api_key("sk-litellm"))
                .with_base_url(Url::parse(base).unwrap()),
        );
        provider
            .language_model("team/gpt-5.6")
            .generate(text_request("hi"))
            .await
            .expect("generate succeeds");
        assert_eq!(
            mock.requests()[0].url.as_str(),
            "http://localhost:4000/litellm/v1/chat/completions",
            "base {base}"
        );
    }
}

#[tokio::test]
async fn ollama_no_auth_sends_no_authorization_header() {
    let mock = MockTransport::shared();
    mock.push_json(200, &chat_ok());
    let provider = provider_with(
        &mock,
        ProviderConfig::openai_chat(Credentials::none())
            .with_base_url(Url::parse("http://localhost:11434/v1").unwrap()),
    );
    provider
        .language_model("qwen3:0.6b")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");
    let http = &mock.requests()[0];
    assert_eq!(
        http.url.as_str(),
        "http://localhost:11434/v1/chat/completions"
    );
    assert!(!http.headers.contains_key("authorization"));
    assert!(!http.headers.contains_key("x-api-key"));
}

#[tokio::test]
async fn bearer_and_custom_header_credentials() {
    let mock = MockTransport::shared();
    mock.push_json(200, &chat_ok());
    let provider = provider_with(
        &mock,
        ProviderConfig::openai_chat(Credentials::bearer("tok"))
            .with_base_url(Url::parse("http://proxy.internal/v1").unwrap()),
    );
    provider
        .language_model("m")
        .generate(text_request("hi"))
        .await
        .unwrap();
    assert_eq!(
        header(&mock.requests()[0], "authorization"),
        Some("Bearer tok")
    );

    let mock = MockTransport::shared();
    mock.push_json(200, &chat_ok());
    let provider = provider_with(
        &mock,
        ProviderConfig::openai_chat(Credentials::header(
            HeaderName::from_static("x-litellm-api-key"),
            "k1",
        ))
        .with_base_url(Url::parse("http://proxy.internal/v1").unwrap()),
    );
    provider
        .language_model("m")
        .generate(text_request("hi"))
        .await
        .unwrap();
    assert!(
        mock.requests()[0]
            .headers
            .iter()
            .any(|(name, value)| name == "x-litellm-api-key" && value == "k1")
    );
}
