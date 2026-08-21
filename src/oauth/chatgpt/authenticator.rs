use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use futures_util::FutureExt;
use futures_util::future::BoxFuture;

use super::device_flow::ChatGptOAuth;
use super::tokens::ChatGptTokens;
use crate::auth::{RequestAuthenticator, TokenStore};
use crate::error::Result;
use crate::oauth::{OAuthAuthenticator, OAuthStatus, OAuthTokens, TokenRefresher};
use crate::transport::HttpRequest;

/// [`RequestAuthenticator`] backed by refreshable ChatGPT tokens.
///
/// Concurrent requests share one refresh. Configure a [`TokenStore`] to save
/// rotating refresh tokens before they become visible to requests. Refreshes
/// only progress while a request awaits them. A cancelled request parks the
/// in-flight refresh until the next request resumes it.
#[must_use = "authenticator modifiers return an updated value"]
pub struct ChatGptAuthenticator {
    inner: OAuthAuthenticator<ChatGptTokens, ChatGptOAuth>,
}

impl fmt::Debug for ChatGptAuthenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ChatGptAuthenticator")
            .field(&self.inner)
            .finish()
    }
}

impl ChatGptAuthenticator {
    /// Create an authenticator from the OAuth client that issued `tokens`.
    pub fn new(tokens: ChatGptTokens, oauth: ChatGptOAuth) -> Self {
        // Refresh a minute early so a token cannot lapse mid-request.
        Self {
            inner: OAuthAuthenticator::new(
                "chatgpt",
                Duration::from_secs(60),
                tokens.derive_metadata(),
                oauth,
            ),
        }
    }

    /// Use the built-in reqwest transport for token refreshes.
    ///
    /// # Errors
    ///
    /// Returns an error if the reqwest transport cannot be initialized, for
    /// example when no Rustls crypto provider is installed (see
    /// [`install_default_crypto_provider`](crate::transport::install_default_crypto_provider)).
    #[cfg(feature = "reqwest-transport")]
    pub fn with_default_transport(tokens: ChatGptTokens) -> Result<Self> {
        Ok(Self::new(tokens, ChatGptOAuth::with_default_transport()?))
    }

    /// Durably save refreshed tokens before publishing them to requests.
    pub fn with_token_store(mut self, store: Arc<dyn TokenStore<ChatGptTokens>>) -> Self {
        self.inner = self.inner.with_token_store(store);
        self
    }

    /// The current published token set.
    pub async fn tokens(&self) -> ChatGptTokens {
        self.inner.tokens().await
    }

    /// Current OAuth credential status.
    pub async fn status(&self) -> OAuthStatus {
        self.inner.status().await
    }
}

#[async_trait::async_trait]
impl RequestAuthenticator for ChatGptAuthenticator {
    async fn authenticate(&self, request: &mut HttpRequest) -> Result<()> {
        self.inner.authenticate(request).await
    }

    async fn reauthenticate(&self, request: &mut HttpRequest) -> Result<bool> {
        self.inner.reauthenticate(request).await
    }
}

impl OAuthTokens for ChatGptTokens {
    fn access_token(&self) -> &str {
        &self.access_token
    }

    fn refresh_token(&self) -> Option<&str> {
        self.refresh_token.as_deref()
    }

    fn expires_at(&self) -> Option<i64> {
        self.expires_at
    }

    fn merge_refreshed(previous: &Self, mut refreshed: Self) -> Self {
        if refreshed.account_id.is_none() {
            refreshed.account_id = previous.account_id.clone();
        }
        if refreshed.id_token.is_none() {
            refreshed.id_token = previous.id_token.clone();
        }
        refreshed
    }

    fn apply(&self, request: &mut HttpRequest) {
        request.set_header("authorization", format!("Bearer {}", self.access_token));
        request
            .headers
            .retain(|(name, _)| !name.eq_ignore_ascii_case("chatgpt-account-id"));
        if let Some(account_id) = &self.account_id {
            request
                .headers
                .push(("chatgpt-account-id".into(), account_id.clone()));
        }
    }
}

impl TokenRefresher<ChatGptTokens> for ChatGptOAuth {
    fn refresh(&self, refresh_token: String) -> BoxFuture<'static, Result<ChatGptTokens>> {
        let oauth = self.clone();
        async move { oauth.refresh(&refresh_token).await }.boxed()
    }
}
