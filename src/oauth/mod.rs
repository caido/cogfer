//! OAuth device-flow sign-in and token refresh.
//!
//! The [`chatgpt`] and [`xai`] clients obtain and refresh subscription tokens
//! and produce a [`RequestAuthenticator`](crate::RequestAuthenticator) for
//! [`ProviderConfig::with_authenticator`](crate::ProviderConfig::with_authenticator).
//! They share [`OAuthClientConfig`] for the client identity, [`DevicePoll`]
//! for device-flow polling, and [`OAuthStatus`] for observing refreshes.
//!
//! Both authenticators are [`OAuthAuthenticator`]s: implement [`OAuthTokens`] and
//! [`TokenRefresher`] for any other refreshable bearer credential (Azure
//! Entra, an STS session) to get the same single-flight refresh policy.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::error::{Error, Result};

pub mod chatgpt;
pub mod xai;

mod authenticator;
mod device_flow;
mod refresh;

pub use self::authenticator::{OAuthAuthenticator, OAuthTokens, TokenRefresher};
#[cfg(feature = "reqwest-transport")]
pub(crate) use self::device_flow::wait_for_device_tokens;
pub(crate) use self::device_flow::{
    auth_error, decode_auth_json, ensure_device_code_is_valid, oauth_error_parts, post_form,
    refresh_error, require_response_field,
};

/// Observable lifecycle of a refreshable OAuth credential.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OAuthStatus {
    /// The current token set can be used.
    Ready,
    /// A refresh is in progress.
    Refreshing,
    /// The last refresh failed transiently and can be retried.
    TransientFailure,
    /// Fresh tokens exist in memory but their durable save failed.
    PersistenceFailed,
    /// The refresh token is permanently unusable. Replace the authenticator
    /// after the user signs in again.
    ReauthRequired,
}

/// Public OAuth client identity used for device authorization and refresh.
///
/// Keep this with the issued tokens because refresh tokens are bound to the
/// client registration that obtained them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthClientConfig {
    client_id: String,
    scope: Option<String>,
}

impl OAuthClientConfig {
    /// Create a client configuration without an explicit OAuth scope.
    ///
    /// # Errors
    ///
    /// Returns an error when `client_id` is empty.
    pub fn new(client_id: impl Into<String>) -> Result<Self> {
        Ok(Self {
            client_id: non_empty("OAuth client id", client_id.into())?,
            scope: None,
        })
    }

    /// Create a client configuration with a space-delimited OAuth scope.
    ///
    /// # Errors
    ///
    /// Returns an error when `client_id` or `scope` is empty.
    pub fn scoped(client_id: impl Into<String>, scope: impl Into<String>) -> Result<Self> {
        Ok(Self {
            client_id: non_empty("OAuth client id", client_id.into())?,
            scope: Some(non_empty("OAuth scope", scope.into())?),
        })
    }

    /// OAuth client identifier.
    pub fn client_id(&self) -> &str {
        &self.client_id
    }

    /// Space-delimited OAuth scope, when the provider requires one.
    pub fn scope(&self) -> Option<&str> {
        self.scope.as_deref()
    }
}

fn non_empty(name: &str, value: String) -> Result<String> {
    let value = value.trim();
    if value.is_empty() {
        return Err(Error::configuration(format!("{name} must not be empty")));
    }
    Ok(value.to_owned())
}

/// The outcome of one device-authorization poll.
#[derive(Debug)]
#[non_exhaustive]
pub enum DevicePoll<T> {
    /// The user has not confirmed yet. Wait for the poll interval and retry.
    Pending,
    /// RFC 8628 `slow_down` with an optional server minimum interval.
    SlowDown { interval: Option<Duration> },
    /// The user confirmed and the tokens are issued.
    Complete(T),
}

pub(crate) fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| i64::try_from(elapsed.as_secs()).unwrap_or(i64::MAX))
        .unwrap_or_default()
}

/// Whether a token expiring at `expires_at` (unix seconds) expires within
/// `skew`. Tokens with an unknown expiry count as fresh: they are refreshed
/// only when the provider rejects them.
pub(crate) fn expires_within(expires_at: Option<i64>, skew: Duration) -> bool {
    let Some(expires_at) = expires_at else {
        return false;
    };
    let skew_seconds = i64::try_from(skew.as_secs()).unwrap_or(i64::MAX);
    unix_now() >= expires_at.saturating_sub(skew_seconds)
}

/// The unix expiry for a token that lives `expires_in` seconds from now.
pub(crate) fn expires_at_from_lifetime(
    provider: &str,
    context: &str,
    expires_in: i64,
) -> Result<i64> {
    if expires_in < 0 {
        return Err(Error::malformed(format!(
            "{provider} {context}: expires_in must not be negative"
        )));
    }
    unix_now().checked_add(expires_in).ok_or_else(|| {
        Error::malformed(format!(
            "{provider} {context}: expires_in exceeds the supported timestamp range"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oauth_client_config_rejects_blank_values() {
        assert!(OAuthClientConfig::new(" ").is_err());
        assert!(OAuthClientConfig::scoped("client", " ").is_err());
    }

    #[test]
    fn expiry_check_honours_skew_and_unknown_expiry() {
        let skew = Duration::from_secs(60);
        assert!(!expires_within(None, skew));
        assert!(expires_within(Some(unix_now() + 30), skew));
        assert!(!expires_within(Some(unix_now() + 3600), skew));
        assert!(expires_within(Some(unix_now() - 10), skew));
    }

    #[test]
    fn token_lifetimes_must_be_representable() {
        assert!(expires_at_from_lifetime("provider", "token refresh", -1).is_err());
        assert!(expires_at_from_lifetime("provider", "token refresh", i64::MAX).is_err());
        assert!(expires_at_from_lifetime("provider", "token refresh", 3600).is_ok());
    }
}
