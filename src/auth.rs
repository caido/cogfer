//! Credentials and the async request-authentication seam.

use std::fmt;

use crate::error::Result;
use crate::transport::HttpRequest;

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

/// A credential string with redacted `Debug` and `Display` output.
///
/// This type deliberately does not implement Serde traits. Persist credentials
/// through an explicit secret-storage boundary instead of serializing this type.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Access the secret value. Use only to construct outgoing requests.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl<T: Into<String>> From<T> for SecretString {
    fn from(value: T) -> Self {
        Self(value.into())
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretString(<redacted>)")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

/// Static credentials for a provider.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Credentials {
    /// No authentication.
    None,
    /// A provider API key, sent in the protocol's native key header
    /// (`Authorization: Bearer` for OpenAI/OpenRouter, `x-api-key` for
    /// Anthropic, `x-goog-api-key` for Gemini).
    ApiKey(SecretString),
    /// An explicit `Authorization: Bearer` token, regardless of protocol.
    Bearer(SecretString),
    /// A custom header.
    Header { name: String, value: SecretString },
}

impl Credentials {
    pub fn api_key(key: impl Into<SecretString>) -> Self {
        Credentials::ApiKey(key.into())
    }

    pub fn bearer(token: impl Into<SecretString>) -> Self {
        Credentials::Bearer(token.into())
    }

    pub fn header(name: impl Into<String>, value: impl Into<SecretString>) -> Self {
        Credentials::Header {
            name: name.into(),
            value: value.into(),
        }
    }

    pub fn none() -> Self {
        Credentials::None
    }

    /// Set this credential on `request` as `Authorization: Bearer` (API keys
    /// and bearer tokens) or as its custom header. `None` leaves the request untouched.
    pub(crate) fn apply_bearer(&self, request: &mut HttpRequest) {
        match self {
            Credentials::None => {}
            Credentials::ApiKey(secret) | Credentials::Bearer(secret) => {
                request.set_header("authorization", format!("Bearer {}", secret.expose()));
            }
            Credentials::Header { name, value } => {
                request.set_header(name, value.expose().to_string());
            }
        }
    }

    /// Set this credential on `request`, sending API keys in `header_name`
    /// (the protocol's native key header) and bearer tokens as
    /// `Authorization: Bearer`. `None` leaves the request untouched.
    pub(crate) fn apply_native_key(&self, request: &mut HttpRequest, header_name: &str) {
        match self {
            Credentials::None => {}
            Credentials::ApiKey(secret) => {
                request.set_header(header_name, secret.expose().to_string());
            }
            Credentials::Bearer(secret) => {
                request.set_header("authorization", format!("Bearer {}", secret.expose()));
            }
            Credentials::Header { name, value } => {
                request.set_header(name, value.expose().to_string());
            }
        }
    }
}

/// Async authentication hook, applied to every outgoing request.
///
/// This is the extension seam for refreshable tokens (e.g. OAuth): implement
/// it to fetch/refresh a token and set the appropriate headers. When a
/// provider is configured with an authenticator, it replaces the static
/// [`Credentials`] header logic. Authentication runs after profile, provider,
/// and per-request headers, so it has final authority over credential headers.
#[async_trait::async_trait]
pub trait RequestAuthenticator: Send + Sync + fmt::Debug {
    /// Authenticate a fully prepared request before it is sent.
    ///
    /// Implementations should replace, not append, credential headers so a
    /// stale caller-supplied value cannot remain active.
    ///
    /// # Errors
    ///
    /// Returns an error when credentials cannot be obtained or refreshed.
    async fn authenticate(&self, request: &mut HttpRequest) -> Result<()>;

    /// Recover once after the provider rejects the prepared credentials.
    ///
    /// Called only for an HTTP 401 received before any response output. The
    /// request contains the rejected credential headers. Return `true` only
    /// after replacing them and when retrying the request once is safe.
    ///
    /// # Errors
    ///
    /// Returns an error when recovery or credential refresh fails.
    async fn reauthenticate(&self, _request: &mut HttpRequest) -> Result<bool> {
        Ok(false)
    }
}

/// Durable storage for refreshed OAuth token sets.
///
/// The host loads the initial token set and passes it to an authenticator.
/// Refreshes await [`TokenStore::save`] before publishing the new tokens to
/// requests. Implementations should reject stale saves after the host has
/// replaced or removed the credential.
#[async_trait::async_trait]
pub trait TokenStore<T>: Send + Sync {
    /// Durably replace the current token set using the store's credential
    /// revision or equivalent stale-write guard.
    ///
    /// # Errors
    ///
    /// Returns an error when the new token set was not durably committed.
    async fn save(&self, tokens: &T) -> Result<()>;
}
