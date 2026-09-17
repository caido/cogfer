#[cfg(feature = "aws")]
use std::sync::Arc;

#[cfg(feature = "aws")]
use cogfer::aws::{AwsCredentials, SigV4Authenticator};
use cogfer::{Credentials, ErrorKind, ProviderConfig};

#[cfg(feature = "aws")]
use crate::common::{
    BEDROCK_REGION, bedrock_config, bedrock_openai_provider_on, bedrock_provider_on,
};
use crate::common::{
    chatgpt_provider_on, live_client, provider_from_env, xai_subscription_provider_on,
};

type Config = fn(Credentials) -> ProviderConfig;

const API_KEY_PROFILES: [(&str, Config); 7] = [
    ("OPENAI_API_KEY", ProviderConfig::openai_responses),
    ("OPENAI_API_KEY", ProviderConfig::openai_chat),
    ("ANTHROPIC_API_KEY", ProviderConfig::anthropic),
    ("GOOGLE_API_KEY", ProviderConfig::gemini),
    ("OPENROUTER_API_KEY", ProviderConfig::openrouter),
    ("XAI_API_KEY", ProviderConfig::xai),
    ("XAI_API_KEY", ProviderConfig::xai_chat),
];

#[tokio::test]
#[ignore = "live"]
async fn api_keys_verify() {
    for (env, config) in API_KEY_PROFILES {
        let Some(provider) = provider_from_env(env, config) else {
            continue;
        };
        provider
            .verify()
            .await
            .unwrap_or_else(|error| panic!("{}: {error}", provider.profile()));
        println!("{} verified", provider.profile());
    }
}

#[tokio::test]
#[ignore = "live"]
async fn wrong_api_keys_are_rejected() {
    let client = live_client();
    for (_, config) in API_KEY_PROFILES {
        let provider = client
            .provider(config(Credentials::api_key("not-a-key")))
            .expect("provider builds");
        let error = provider
            .verify()
            .await
            .expect_err("a wrong key must not verify");
        assert_eq!(
            error.kind(),
            ErrorKind::Authentication,
            "{}: {error}",
            provider.profile()
        );
        println!("{} rejected: {error}", provider.profile());
    }
}

#[cfg(feature = "aws")]
#[tokio::test]
#[ignore = "live"]
async fn bedrock_credentials_verify() {
    let client = live_client();
    if let Some(provider) = bedrock_provider_on(&client) {
        provider.verify().await.expect("a Bedrock API key verifies");
        println!("bedrock api key verified");
    }
    if let Some(provider) = bedrock_openai_provider_on(&client) {
        provider
            .verify()
            .await
            .expect("the OpenAI profile verifies with the same key");
        println!("bedrock openai profile verified");
    }
    if let Some(provider) = crate::live::bedrock::sigv4_provider() {
        provider.verify().await.expect("SigV4 credentials verify");
        println!("bedrock sigv4 verified");
    }
}

#[cfg(feature = "aws")]
#[tokio::test]
#[ignore = "live"]
async fn wrong_bedrock_credentials_are_rejected() {
    let client = live_client();

    let bearer = client
        .provider(bedrock_config(Credentials::bearer(
            "bedrock-api-key-not-a-key",
        )))
        .expect("provider builds");
    let error = bearer
        .verify()
        .await
        .expect_err("a wrong key must not verify");
    assert_eq!(error.kind(), ErrorKind::Authentication, "{error}");
    println!("bedrock api key rejected: {error}");

    let credentials = AwsCredentials::new("AKIAEXAMPLE", "not-a-secret", None, None, "test");
    let signed = client
        .provider(
            bedrock_config(Credentials::none()).with_authenticator(Arc::new(
                SigV4Authenticator::new(BEDROCK_REGION, credentials),
            )),
        )
        .expect("provider builds");
    let error = signed
        .verify()
        .await
        .expect_err("a wrong secret must not verify");
    assert_eq!(error.kind(), ErrorKind::Authentication, "{error}");
    println!("bedrock sigv4 rejected: {error}");
}

#[tokio::test]
#[ignore = "live"]
async fn subscriptions_verify() {
    let client = live_client();
    if let Some(provider) = chatgpt_provider_on(&client) {
        provider
            .verify()
            .await
            .expect("the ChatGPT subscription verifies");
        println!("chatgpt subscription verified");
    }
    if let Some(provider) = xai_subscription_provider_on(&client, ProviderConfig::xai) {
        provider
            .verify()
            .await
            .expect("the xAI subscription verifies");
        println!("xai subscription verified");
    }
}
