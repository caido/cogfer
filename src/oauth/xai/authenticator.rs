use std::time::Duration;

use futures_util::FutureExt;
use futures_util::future::BoxFuture;

use super::{XaiOAuth, XaiTokens};
use crate::error::Result;
use crate::http::bearer_value;
use crate::oauth::{OAuthAuthenticator, OAuthTokens, TokenRefresher};
use crate::transport::{HttpRequest, header};

/// [`RequestAuthenticator`](crate::RequestAuthenticator) backed by refreshable
/// SpaceXAI tokens. See [`OAuthAuthenticator`] for the refresh policy.
pub type XaiAuthenticator = OAuthAuthenticator<XaiTokens, XaiOAuth>;

impl XaiAuthenticator {
    /// Use the built-in reqwest transport for token refreshes.
    ///
    /// # Errors
    ///
    /// Returns an error if the reqwest transport cannot be initialized, for
    /// example when no Rustls crypto provider is installed (see
    /// [`install_default_crypto_provider`](crate::transport::install_default_crypto_provider)).
    #[cfg(feature = "reqwest-transport")]
    pub fn with_default_transport(tokens: XaiTokens) -> Result<Self> {
        Ok(Self::new(tokens, XaiOAuth::with_default_transport()?))
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
    fn apply(&self, request: &mut HttpRequest) -> Result<()> {
        request
            .headers
            .insert(header::AUTHORIZATION, bearer_value(&self.access_token)?);
        Ok(())
    }
}

impl TokenRefresher<XaiTokens> for XaiOAuth {
    const PROVIDER: &'static str = "xai";
    /// SpaceXAI tokens live about an hour, so refresh five minutes early.
    const EXPIRY_SKEW: Duration = Duration::from_secs(5 * 60);

    fn refresh(&self, refresh_token: String) -> BoxFuture<'static, Result<XaiTokens>> {
        let oauth = self.clone();
        async move { oauth.refresh(&refresh_token).await }.boxed()
    }
}
