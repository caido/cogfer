use std::time::Duration;

use futures_util::FutureExt;
use futures_util::future::BoxFuture;

use super::device_flow::ChatGptOAuth;
use super::tokens::ChatGptTokens;
use crate::error::Result;
use crate::http::{bearer_value, header_value};
use crate::oauth::{OAuthAuthenticator, OAuthTokens, TokenRefresher};
use crate::transport::{HeaderName, HttpRequest, header};

/// [`RequestAuthenticator`](crate::RequestAuthenticator) backed by refreshable
/// ChatGPT tokens. See [`OAuthAuthenticator`] for the refresh policy.
pub type ChatGptAuthenticator = OAuthAuthenticator<ChatGptTokens, ChatGptOAuth>;

impl ChatGptAuthenticator {
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
}

const CHATGPT_ACCOUNT_ID: HeaderName = HeaderName::from_static("chatgpt-account-id");

impl OAuthTokens for ChatGptTokens {
    const PROVIDER: &'static str = "chatgpt";
    const EXPIRY_SKEW: Duration = Duration::from_secs(60);

    fn access_token(&self) -> &str {
        &self.access_token
    }

    fn refresh_token(&self) -> Option<&str> {
        self.refresh_token.as_deref()
    }

    fn expires_at(&self) -> Option<i64> {
        self.expires_at
    }

    fn normalized(self) -> Self {
        self.derive_metadata()
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

    fn apply(&self, request: &mut HttpRequest) -> Result<()> {
        request
            .headers
            .insert(header::AUTHORIZATION, bearer_value(&self.access_token)?);
        request.headers.remove(CHATGPT_ACCOUNT_ID);
        if let Some(account_id) = &self.account_id {
            request
                .headers
                .insert(CHATGPT_ACCOUNT_ID, header_value(account_id)?);
        }
        Ok(())
    }
}

impl TokenRefresher<ChatGptTokens> for ChatGptOAuth {
    fn refresh(&self, refresh_token: String) -> BoxFuture<'static, Result<ChatGptTokens>> {
        let oauth = self.clone();
        async move { oauth.refresh(&refresh_token).await }.boxed()
    }
}
