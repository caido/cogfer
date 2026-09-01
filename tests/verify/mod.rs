//! `Provider::verify`: the free credential check each profile sends, and how
//! its answers are classified.

use std::sync::Arc;

use llmwire::transport::mock::MockTransport;
use llmwire::transport::{HeaderName, HeaderValue, HttpRequest, Method};
use llmwire::{Credentials, ErrorKind, Provider, ProviderConfig};
use serde_json::json;
use url::Url;

use crate::common::{
    anthropic, chatgpt, gemini, header, headers, openai_chat, openai_responses, openrouter,
    provider_with, xai, xai_chat,
};

/// Verifies against a canned `200` and hands back the request that was sent.
async fn verified_request(provider: Provider, mock: &MockTransport) -> HttpRequest {
    mock.push_json(200, &json!({"object": "list", "data": []}));
    provider.verify().await.expect("valid credentials verify");

    let requests = mock.requests();
    assert_eq!(requests.len(), 1);
    let request = requests.into_iter().next().unwrap();
    assert_eq!(request.method, Method::GET);
    assert!(request.body.is_none());
    assert_eq!(header(&request, "accept"), Some("application/json"));
    request
}

#[tokio::test]
async fn openai_profiles_list_models_with_the_bearer_key() {
    for (build, url) in [
        (
            openai_responses as fn(&Arc<MockTransport>) -> Provider,
            "https://api.openai.com/v1/models",
        ),
        (openai_chat, "https://api.openai.com/v1/models"),
        (xai, "https://api.x.ai/v1/models"),
        (xai_chat, "https://api.x.ai/v1/models"),
    ] {
        let mock = MockTransport::shared();
        let request = verified_request(build(&mock), &mock).await;
        assert_eq!(request.url.as_str(), url);
        let authorization = header(&request, "authorization").expect("bearer key");
        assert!(
            authorization.starts_with("Bearer "),
            "{url}: {authorization}"
        );
    }
}

#[tokio::test]
async fn openrouter_describes_the_key() {
    let mock = MockTransport::shared();
    let request = verified_request(openrouter(&mock), &mock).await;
    assert_eq!(request.url.as_str(), "https://openrouter.ai/api/v1/key");
    assert_eq!(header(&request, "authorization"), Some("Bearer sk-or-test"));
}

#[tokio::test]
async fn anthropic_lists_models_on_the_versioned_api() {
    let mock = MockTransport::shared();
    let request = verified_request(anthropic(&mock), &mock).await;
    assert_eq!(request.url.as_str(), "https://api.anthropic.com/v1/models");
    assert_eq!(header(&request, "x-api-key"), Some("sk-ant-test"));
    assert_eq!(header(&request, "anthropic-version"), Some("2023-06-01"));
    assert!(header(&request, "authorization").is_none());
}

#[tokio::test]
async fn gemini_lists_models_with_the_google_key() {
    let mock = MockTransport::shared();
    let request = verified_request(gemini(&mock), &mock).await;
    assert_eq!(
        request.url.as_str(),
        "https://generativelanguage.googleapis.com/v1beta/models"
    );
    assert_eq!(header(&request, "x-goog-api-key"), Some("g-test"));
}

#[tokio::test]
async fn chatgpt_lists_models_for_its_client_version() {
    let mock = MockTransport::shared();
    let request = verified_request(chatgpt(&mock), &mock).await;
    assert_eq!(
        request.url.as_str(),
        format!(
            "https://chatgpt.com/backend-api/codex/models?client_version={}",
            env!("CARGO_PKG_VERSION")
        )
    );
    assert_eq!(
        header(&request, "authorization"),
        Some("Bearer chatgpt-access-token")
    );
}

#[tokio::test]
async fn custom_base_urls_keep_their_prefix_and_provider_headers() {
    let mock = MockTransport::shared();
    let provider = provider_with(
        &mock,
        ProviderConfig::openai_chat(Credentials::none())
            .with_base_url(Url::parse("http://localhost:4000/litellm/v1/").unwrap())
            .with_header(
                HeaderName::from_static("x-title"),
                HeaderValue::from_static("caido"),
            ),
    );

    let request = verified_request(provider, &mock).await;

    assert_eq!(
        request.url.as_str(),
        "http://localhost:4000/litellm/v1/models"
    );
    assert_eq!(header(&request, "x-title"), Some("caido"));
    assert!(header(&request, "authorization").is_none());
}

#[tokio::test]
async fn rejected_credentials_are_authentication_failures() {
    let mock = MockTransport::shared();
    mock.push_json(
        401,
        &json!({"error": {
            "message": "Incorrect API key provided",
            "type": "invalid_request_error",
            "code": "invalid_api_key"
        }}),
    );

    let error = openai_responses(&mock).verify().await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Authentication);
    assert_eq!(error.status(), Some(401));
    assert_eq!(error.code(), Some("invalid_api_key"));
    assert_eq!(error.origin(), Some("openai-responses"));
    assert_eq!(error.message(), "Incorrect API key provided");
}

#[tokio::test]
async fn a_route_the_server_does_not_serve_is_not_found() {
    let mock = MockTransport::shared();
    mock.push_response(404, headers(&[]), "no route");

    let error = openai_chat(&mock).verify().await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(error.status(), Some(404));
}

#[tokio::test]
async fn transport_failures_are_attributed_to_the_profile() {
    let mock = MockTransport::shared();

    let error = gemini(&mock).verify().await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Transport);
    assert_eq!(error.origin(), Some("gemini"));
}
