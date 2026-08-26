use std::time::Duration;

use futures_util::FutureExt;
use futures_util::future::BoxFuture;

use super::{XaiOAuth, XaiTokens};
use crate::error::Result;
use crate::http::bearer_value;
use crate::oauth::{OAuthAuthenticator, OAuthTokens, TokenRefresher};
use crate::transport::{HttpRequest, header};

/// [`RequestAuthenticator`](crate::RequestAuthenticator) backed by refreshable
/// xAI tokens. See [`OAuthAuthenticator`] for the refresh policy.
pub type XaiAuthenticator = OAuthAuthenticator<XaiTokens, XaiOAuth>;

impl OAuthTokens for XaiTokens {
    const PROVIDER: &'static str = "xai";
    /// xAI tokens live about an hour, so refresh five minutes early.
    const EXPIRY_SKEW: Duration = Duration::from_secs(5 * 60);

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
    fn refresh(&self, refresh_token: String) -> BoxFuture<'static, Result<XaiTokens>> {
        let oauth = self.clone();
        async move { oauth.refresh(&refresh_token).await }.boxed()
    }

    #[cfg(feature = "reqwest-transport")]
    fn with_default_transport() -> Result<Self> {
        XaiOAuth::with_default_transport()
    }
}
