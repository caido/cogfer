//! OpenAI models on Amazon Bedrock's OpenAI-compatible Responses API, over
//! both credentials the service accepts: a Bedrock API key in
//! `AWS_BEARER_TOKEN_BEDROCK`, and SigV4 signing from `AWS_ACCESS_KEY_ID` and
//! `AWS_SECRET_ACCESS_KEY`.
//!
//! Two backends serve the wire format. The `bedrock-runtime` tests target
//! [`MODEL`], a frontier GPT inference profile, and need OpenAI GPT model
//! access on the account (Bedrock answers `access_denied` pointing at AWS
//! Sales without it). The `bedrock-mantle` tests reach the same profile
//! through a custom base URL with [`MANTLE_MODEL`], which plain gpt-oss
//! access covers, so they exercise the full codec on any account.

use std::sync::Arc;

use llmwire::aws::{AwsCredentials, SigV4Authenticator};
use llmwire::{Credentials, Provider, ReasoningConfig, ReasoningEffort, StreamEvent};
use url::Url;

use crate::common::{
    BEDROCK_REGION, agentic_loop, assert_terminal_contract, bedrock_openai_config,
    bedrock_openai_provider_on, drain, live_client, loop_request, provider_from_env_on,
    text_request,
};

/// The frontier GPT models are served only as cross-region inference
/// profiles, so the model id carries the `us.` prefix.
pub(crate) const MODEL: &str = "us.openai.gpt-5.6-sol";

/// gpt-oss speaks the Responses API only on `bedrock-mantle`; on
/// `bedrock-runtime` it is limited to Chat Completions.
pub(crate) const MANTLE_MODEL: &str = "openai.gpt-oss-120b";

const MANTLE_BASE_URL: &str = "https://bedrock-mantle.us-west-2.api.aws/v1";

/// A `bedrock-runtime` provider authenticating with a Bedrock API key.
pub(crate) fn bearer_provider() -> Option<Provider> {
    bedrock_openai_provider_on(&live_client())
}

/// A `bedrock-mantle` provider from the same Bedrock API key.
pub(crate) fn mantle_provider() -> Option<Provider> {
    provider_from_env_on(&live_client(), "AWS_BEARER_TOKEN_BEDROCK", |credentials| {
        bedrock_openai_config(credentials)
            .with_base_url(Url::parse(MANTLE_BASE_URL).expect("mantle base URL is valid"))
    })
}

/// A `bedrock-runtime` provider signing each request with SigV4.
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
                bedrock_openai_config(Credentials::none()).with_authenticator(Arc::new(
                    SigV4Authenticator::new(BEDROCK_REGION, credentials),
                )),
            )
            .expect("provider builds"),
    )
}

/// The blocking path on mantle: the Responses body, `store: false`, and the
/// encrypted-reasoning include against a real Bedrock server.
#[tokio::test]
#[ignore = "live"]
async fn bedrock_openai_mantle_generates_text() {
    let Some(provider) = mantle_provider() else {
        return;
    };
    let result = provider
        .language_model(MANTLE_MODEL)
        .generate(text_request("Say OK."))
        .await
        .expect("generate succeeds");

    assert!(!result.text().trim().is_empty(), "{result:?}");
    assert!(result.usage.output_tokens.is_some(), "{result:?}");
}

/// The SSE Responses stream on mantle.
#[tokio::test]
#[ignore = "live"]
async fn bedrock_openai_mantle_streams_text() {
    let Some(provider) = mantle_provider() else {
        return;
    };
    let events = drain(
        provider
            .language_model(MANTLE_MODEL)
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

/// A tool loop with reasoning on mantle, the shape a real agent uses. The
/// second turn replays the first turn's reasoning items, so this covers
/// stateless reasoning replay with `store: false`.
#[tokio::test]
#[ignore = "live"]
async fn bedrock_openai_mantle_agentic_loop_with_effort() {
    let Some(provider) = mantle_provider() else {
        return;
    };
    let mut request = loop_request();
    request.reasoning = Some(ReasoningConfig::effort(ReasoningEffort::Low));
    agentic_loop(&provider.language_model(MANTLE_MODEL), request).await;
}

/// The blocking path on `bedrock-runtime` over an API key. Needs OpenAI GPT
/// model access on the account.
#[tokio::test]
#[ignore = "live"]
async fn bedrock_openai_bearer_generates_text() {
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

/// The SSE Responses stream on `bedrock-runtime` over a SigV4 signature.
/// Needs OpenAI GPT model access on the account.
#[tokio::test]
#[ignore = "live"]
async fn bedrock_openai_sigv4_streams_text() {
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
