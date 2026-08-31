//! Helpers for tests that talk to real providers: the ignored live tests and
//! cassette recording.

use std::path::PathBuf;
use std::sync::Arc;

use llmwire::oauth::chatgpt::{ChatGptAuthenticator, ChatGptTokens};
use llmwire::oauth::xai::{XaiAuthenticator, XaiTokens};
use llmwire::{Client, Credentials, Provider, ProviderConfig, TokenStore};

pub(crate) fn live_client() -> Client {
    let _ = dotenvy::dotenv();
    llmwire::transport::install_default_crypto_provider();
    Client::builder().build().expect("client builds")
}

/// The API key in `env`, or `None` (with a SKIP note) when it is not configured.
fn credentials_from_env(env: &str) -> Option<Credentials> {
    let _ = dotenvy::dotenv();
    match std::env::var(env) {
        Ok(key) => Some(Credentials::api_key(key)),
        Err(_) => {
            eprintln!("SKIP: {env} not set");
            None
        }
    }
}

/// A provider on `client` for the API key in `env`, or `None` (with a SKIP
/// note) when the key is not configured.
pub(crate) fn provider_from_env_on(
    client: &Client,
    env: &str,
    config: impl FnOnce(Credentials) -> ProviderConfig,
) -> Option<Provider> {
    let credentials = credentials_from_env(env)?;
    Some(
        client
            .provider(config(credentials))
            .expect("provider builds"),
    )
}

/// [`provider_from_env_on`] with the default live client.
pub(crate) fn provider_from_env(
    env: &str,
    config: impl FnOnce(Credentials) -> ProviderConfig,
) -> Option<Provider> {
    provider_from_env_on(&live_client(), env, config)
}

/// A Bedrock provider on `client` from the `AWS_BEARER_TOKEN_BEDROCK` Bedrock
/// API key, or `None` (with a SKIP note) when it is not configured.
///
/// Bedrock API keys are bearer tokens, so this needs no SigV4 signing. The
/// live tests cover the SigV4 path separately.
pub(crate) fn bedrock_provider_on(client: &Client) -> Option<Provider> {
    provider_from_env_on(client, "AWS_BEARER_TOKEN_BEDROCK", super::bedrock_config)
}

/// A ChatGPT subscription provider on `client`, from `CHATGPT_ACCESS_TOKEN`
/// (plus optional `CHATGPT_REFRESH_TOKEN` / `CHATGPT_ACCOUNT_ID`) or the
/// sign-in cached by `cargo run --example chatgpt_login`. Refreshed tokens are
/// written back to the cache. `None` (with a SKIP note) when neither exists.
pub(crate) fn chatgpt_provider_on(client: &Client) -> Option<Provider> {
    let _ = dotenvy::dotenv();
    let (tokens, cache_path) = if let Ok(access_token) = std::env::var("CHATGPT_ACCESS_TOKEN") {
        let mut tokens = ChatGptTokens::new(access_token);
        if let Ok(refresh_token) = std::env::var("CHATGPT_REFRESH_TOKEN") {
            tokens = tokens.with_refresh_token(refresh_token);
        }
        if let Ok(account_id) = std::env::var("CHATGPT_ACCOUNT_ID") {
            tokens = tokens.with_account_id(account_id);
        }
        (tokens, None)
    } else if let Some((tokens, path)) = cached_tokens("chatgpt.json") {
        (tokens, Some(path))
    } else {
        eprintln!(
            "SKIP: no CHATGPT_ACCESS_TOKEN and no cached sign-in \
             (run `cargo run --example chatgpt_login`)"
        );
        return None;
    };

    // The authenticator keeps its own transport so token exchanges never
    // pass through the client's (possibly recording) transport.
    let mut authenticator =
        ChatGptAuthenticator::with_default_transport(tokens).expect("authenticator builds");
    if let Some(path) = cache_path {
        authenticator = authenticator.with_token_store(Arc::new(JsonTokenStore { path }));
    }
    Some(
        client
            .provider(
                ProviderConfig::chatgpt(Credentials::none())
                    .with_authenticator(Arc::new(authenticator)),
            )
            .expect("provider builds"),
    )
}

/// A SpaceXAI provider on `client` for `config` (Responses or Chat dialect),
/// from `XAI_API_KEY`, `XAI_ACCESS_TOKEN` (plus optional `XAI_REFRESH_TOKEN`),
/// or the sign-in cached by `cargo run --example xai_login`. `None` (with a
/// SKIP note) when none exists.
pub(crate) fn xai_provider_on(
    client: &Client,
    config: impl FnOnce(Credentials) -> ProviderConfig,
) -> Option<Provider> {
    let _ = dotenvy::dotenv();
    if let Ok(api_key) = std::env::var("XAI_API_KEY") {
        return Some(
            client
                .provider(config(Credentials::api_key(api_key)))
                .expect("provider builds"),
        );
    }
    let (tokens, cache_path) = if let Ok(access_token) = std::env::var("XAI_ACCESS_TOKEN") {
        let mut tokens = XaiTokens::new(access_token);
        if let Ok(refresh_token) = std::env::var("XAI_REFRESH_TOKEN") {
            tokens = tokens.with_refresh_token(refresh_token);
        }
        (tokens, None)
    } else if let Some((tokens, path)) = cached_tokens("xai.json") {
        (tokens, Some(path))
    } else {
        eprintln!(
            "SKIP: no XAI_API_KEY/XAI_ACCESS_TOKEN and no cached sign-in \
             (run `cargo run --example xai_login`)"
        );
        return None;
    };

    let mut authenticator =
        XaiAuthenticator::with_default_transport(tokens).expect("authenticator builds");
    if let Some(path) = cache_path {
        authenticator = authenticator.with_token_store(Arc::new(JsonTokenStore { path }));
    }
    Some(
        client
            .provider(config(Credentials::none()).with_authenticator(Arc::new(authenticator)))
            .expect("provider builds"),
    )
}

/// Tokens cached by the login examples under the user config directory.
fn cached_tokens<T: serde::de::DeserializeOwned>(file: &str) -> Option<(T, PathBuf)> {
    #[cfg(target_os = "windows")]
    let base = std::env::var_os("APPDATA").map(PathBuf::from);
    #[cfg(not(target_os = "windows"))]
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")));
    let path = base?.join("llmwire").join(file);
    let tokens = serde_json::from_slice(&std::fs::read(&path).ok()?).ok()?;
    Some((tokens, path))
}

/// Writes refreshed tokens back to the login example's cache file.
struct JsonTokenStore {
    path: PathBuf,
}

#[async_trait::async_trait]
impl TokenStore<ChatGptTokens> for JsonTokenStore {
    async fn save(&self, tokens: &ChatGptTokens) -> llmwire::Result<()> {
        save_tokens(&self.path, tokens)
    }
}

#[async_trait::async_trait]
impl TokenStore<XaiTokens> for JsonTokenStore {
    async fn save(&self, tokens: &XaiTokens) -> llmwire::Result<()> {
        save_tokens(&self.path, tokens)
    }
}

fn save_tokens(path: &std::path::Path, tokens: &impl serde::Serialize) -> llmwire::Result<()> {
    let bytes = serde_json::to_vec_pretty(tokens).map_err(|error| {
        llmwire::Error::new(
            llmwire::ErrorKind::Provider,
            format!("failed to serialize refreshed tokens: {error}"),
        )
    })?;
    std::fs::write(path, bytes).map_err(|error| {
        llmwire::Error::new(
            llmwire::ErrorKind::Provider,
            format!("failed to persist refreshed tokens: {error}"),
        )
    })
}
