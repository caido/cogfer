//! Anthropic on Amazon Bedrock, over both credentials the service accepts: a
//! Bedrock API key in `AWS_BEARER_TOKEN_BEDROCK`, and SigV4 signing from
//! `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and an optional
//! `AWS_SESSION_TOKEN`.
//!
//! Both run against [`BEDROCK_REGION`] rather than `AWS_REGION`, because
//! [`MODEL`] is a US inference profile and the cassettes are recorded there.

use std::sync::Arc;

use cogfer::aws::{AwsCredentials, SigV4Authenticator};
use cogfer::{Credentials, Provider, ReasoningConfig, ReasoningEffort, StreamEvent};

use crate::common::{
    BEDROCK_REGION, agentic_loop, assert_terminal_contract, bedrock_config, bedrock_provider_on,
    drain, live_client, loop_request, text_request,
};

/// Bedrock serves Anthropic models only through cross-region inference
/// profiles, so the model id carries the `us.` prefix rather than being the
/// bare `anthropic.` id an on-demand invocation would use.
pub(crate) const MODEL: &str = "us.anthropic.claude-sonnet-4-6";

/// A provider authenticating with a Bedrock API key, the credential the
/// cassettes are recorded with.
pub(crate) fn bearer_provider() -> Option<Provider> {
    bedrock_provider_on(&live_client())
}

/// A provider signing each request with SigV4, the other credential Bedrock
/// accepts.
pub(crate) fn sigv4_provider() -> Option<Provider> {
    let _ = dotenvy::dotenv();
    let (Ok(access_key_id), Ok(secret)) = (
        std::env::var("AWS_ACCESS_KEY_ID"),
        std::env::var("AWS_SECRET_ACCESS_KEY"),
    ) else {
        eprintln!("SKIP: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set");
        return None;
    };
    let credentials = AwsCredentials::new(
        access_key_id,
        secret,
        std::env::var("AWS_SESSION_TOKEN").ok(),
        None,
        "environment",
    );
    Some(
        live_client()
            .provider(
                bedrock_config(Credentials::none()).with_authenticator(Arc::new(
                    SigV4Authenticator::new(BEDROCK_REGION, credentials),
                )),
            )
            .expect("provider builds"),
    )
}

/// The blocking path over an API key: Bedrock's `invoke` endpoint and the
/// Anthropic body it wraps.
#[tokio::test]
#[ignore = "live"]
async fn bedrock_bearer_generates_text() {
    let Some(provider) = bearer_provider() else {
        return;
    };
    let result = provider
        .language_model(MODEL)
        .generate(text_request("Say OK."))
        .await
        .expect("generate succeeds");

    assert!(!result.text().trim().is_empty(), "{result:?}");
    assert!(result.usage.output_tokens.is_some(), "{result:?}");
}

/// The AWS event stream decoder over an API key.
#[tokio::test]
#[ignore = "live"]
async fn bedrock_bearer_streams_text() {
    let Some(provider) = bearer_provider() else {
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

/// A tool loop with reasoning over an API key, the shape a real agent uses.
#[tokio::test]
#[ignore = "live"]
async fn bedrock_bearer_agentic_loop_with_effort() {
    let Some(provider) = bearer_provider() else {
        return;
    };
    let mut request = loop_request();
    request.reasoning = Some(ReasoningConfig::effort(ReasoningEffort::Low));
    agentic_loop(&provider.language_model(MODEL), request).await;
}

/// The AWS event stream decoder over a SigV4 signature.
#[tokio::test]
#[ignore = "live"]
async fn bedrock_sigv4_streams_text() {
    let Some(provider) = sigv4_provider() else {
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
async fn bedrock_sigv4_agentic_loop_with_effort() {
    let Some(provider) = sigv4_provider() else {
        return;
    };
    let mut request = loop_request();
    request.reasoning = Some(ReasoningConfig::effort(ReasoningEffort::Low));
    agentic_loop(&provider.language_model(MODEL), request).await;
}
