use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use futures_util::FutureExt;
use futures_util::future::BoxFuture;

use super::{XaiOAuth, XaiTokens};
use crate::auth::{OAuthStatus, RequestAuthenticator, TokenStore};
use crate::error::Result;
use crate::oauth::{OAuthAuthenticator, OAuthTokens, TokenRefresher};
use crate::transport::HttpRequest;

/// [`RequestAuthenticator`] backed by refreshable xAI tokens.
///
/// Concurrent requests share one refresh. Configure a [`TokenStore`] to save
/// rotating refresh tokens before they become visible to requests. Refreshes
/// only progress while a request awaits them. A cancelled request parks the
/// in-flight refresh until the next request resumes it.
#[must_use = "authenticator modifiers return an updated value"]
pub struct XaiAuthenticator {
    inner: OAuthAuthenticator<XaiTokens, XaiOAuth>,
}

impl fmt::Debug for XaiAuthenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("XaiAuthenticator")
            .field(&self.inner)
            .finish()
    }
}

impl XaiAuthenticator {
    /// Create an authenticator from the OAuth client that issued `tokens`.
    pub fn new(tokens: XaiTokens, oauth: XaiOAuth) -> Self {
        // xAI tokens live about an hour, so refresh five minutes early.
        Self {
            inner: OAuthAuthenticator::new("xai", Duration::from_secs(5 * 60), tokens, oauth),
        }
    }

    /// Use the built-in reqwest transport for token refreshes.
    ///
    /// # Errors
    ///
    /// Returns an error if the reqwest transport cannot be initialized.
    #[cfg(feature = "reqwest-transport")]
    pub fn with_default_transport(tokens: XaiTokens) -> Result<Self> {
        Ok(Self::new(tokens, XaiOAuth::with_default_transport()?))
    }

    /// Durably save refreshed tokens before publishing them to requests.
    pub fn with_token_store(mut self, store: Arc<dyn TokenStore<XaiTokens>>) -> Self {
        self.inner = self.inner.with_token_store(store);
        self
    }

    /// The current published token set.
    pub async fn tokens(&self) -> XaiTokens {
        self.inner.tokens().await
    }

    /// Current OAuth credential status.
    pub async fn status(&self) -> OAuthStatus {
        self.inner.status().await
    }
}

#[async_trait::async_trait]
impl RequestAuthenticator for XaiAuthenticator {
    async fn authenticate(&self, request: &mut HttpRequest) -> Result<()> {
        self.inner.authenticate(request).await
    }

    async fn reauthenticate(&self, request: &mut HttpRequest) -> Result<bool> {
        self.inner.reauthenticate(request).await
    }
}

impl OAuthTokens for XaiTokens {
    fn access_token(&self) -> &str {
        &self.access_token
    }

    fn refresh_token(&self) -> Option<&str> {
        self.refresh_token.as_deref()
    }

    fn expires_at(&self) -> Option<i64> {
        self.expires_at
    }

    fn merge_refreshed(_previous: &Self, refreshed: Self) -> Self {
        refreshed
    }

    /// The OAuth access token acts as a plain API key on `api.x.ai`.
    fn apply(&self, request: &mut HttpRequest) {
        request.set_header("authorization", format!("Bearer {}", self.access_token));
    }
}

impl TokenRefresher<XaiTokens> for XaiOAuth {
    fn refresh(&self, refresh_token: String) -> BoxFuture<'static, Result<XaiTokens>> {
        let oauth = self.clone();
        async move { oauth.refresh(&refresh_token).await }.boxed()
    }
}
