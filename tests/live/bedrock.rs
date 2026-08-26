//! Anthropic on Amazon Bedrock, signed with SigV4 from `AWS_ACCESS_KEY_ID`,
//! `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`, and `AWS_REGION`.

use std::sync::Arc;

use caido_ai::aws::{AwsCredentials, SigV4Authenticator};
use caido_ai::{
    Credentials, Provider, ProviderConfig, ReasoningConfig, ReasoningEffort, StreamEvent,
};

use crate::common::{
    agentic_loop, assert_terminal_contract, drain, live_client, loop_request, text_request,
};

const MODEL: &str = "anthropic.claude-haiku-4-5-20251001-v1:0";

fn provider_from_env() -> Option<Provider> {
    let _ = dotenvy::dotenv();
    let (Ok(access_key_id), Ok(secret)) = (
        std::env::var("AWS_ACCESS_KEY_ID"),
        std::env::var("AWS_SECRET_ACCESS_KEY"),
    ) else {
        eprintln!("SKIP: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set");
        return None;
    };
    let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".into());
    let mut credentials = AwsCredentials::new(access_key_id, secret);
    if let Ok(token) = std::env::var("AWS_SESSION_TOKEN") {
        credentials = credentials.with_session_token(token);
    }
    Some(
        live_client()
            .provider(
                ProviderConfig::bedrock_anthropic(&region, Credentials::none())
                    .expect("region is valid")
                    .with_authenticator(Arc::new(SigV4Authenticator::new(&region, credentials))),
            )
            .expect("provider builds"),
    )
}

#[tokio::test]
#[ignore = "live"]
async fn bedrock_streaming_text() {
    let Some(provider) = provider_from_env() else {
        return;
    };
    let events = drain(
        provider
            .language_model(MODEL)
            .stream(text_request("Say OK."))
            .await
            .expect("stream establishes"),
    )
    .await;

    assert_terminal_contract(&events);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta { .. }))
    );
}

#[tokio::test]
#[ignore = "live"]
async fn bedrock_agentic_loop_with_effort() {
    let Some(provider) = provider_from_env() else {
        return;
    };
    let mut request = loop_request();
    request.reasoning = Some(ReasoningConfig::effort(ReasoningEffort::Low));
    agentic_loop(&provider.language_model(MODEL), request).await;
}
