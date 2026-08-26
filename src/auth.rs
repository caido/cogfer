//! Credentials and the async request-authentication seam.

use std::fmt;

use http::header::{self, HeaderName};

use crate::error::Result;
use crate::http::{bearer_value, header_value};
use crate::transport::HttpRequest;

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
    Header {
        name: HeaderName,
        value: SecretString,
    },
}

impl Credentials {
    pub fn api_key(key: impl Into<SecretString>) -> Self {
        Credentials::ApiKey(key.into())
    }

    pub fn bearer(token: impl Into<SecretString>) -> Self {
        Credentials::Bearer(token.into())
    }

    pub fn header(name: HeaderName, value: impl Into<SecretString>) -> Self {
        Credentials::Header {
            name,
            value: value.into(),
        }
    }

    pub fn none() -> Self {
        Credentials::None
    }

    /// Set this credential on `request` as `Authorization: Bearer` (API keys
    /// and bearer tokens) or as its custom header. `None` leaves the request untouched.
    ///
    /// # Errors
    ///
    /// Returns an error when the credential is not a valid header value.
    pub(crate) fn apply_bearer(&self, request: &mut HttpRequest) -> Result<()> {
        match self {
            Credentials::None => {}
            Credentials::ApiKey(secret) | Credentials::Bearer(secret) => {
                request
                    .headers
                    .insert(header::AUTHORIZATION, bearer_value(secret.expose())?);
            }
            Credentials::Header { name, value } => {
                request.headers.insert(name.clone(), secret_value(value)?);
            }
        }
        Ok(())
    }

    /// Set this credential on `request`, sending API keys in `header_name`
    /// (the protocol's native key header) and bearer tokens as
    /// `Authorization: Bearer`. `None` leaves the request untouched.
    ///
    /// # Errors
    ///
    /// Returns an error when the credential is not a valid header value.
    pub(crate) fn apply_native_key(
        &self,
        request: &mut HttpRequest,
        header_name: HeaderName,
    ) -> Result<()> {
        match self {
            Credentials::None => {}
            Credentials::ApiKey(secret) => {
                request.headers.insert(header_name, secret_value(secret)?);
            }
            Credentials::Bearer(secret) => {
                request
                    .headers
                    .insert(header::AUTHORIZATION, bearer_value(secret.expose())?);
            }
            Credentials::Header { name, value } => {
                request.headers.insert(name.clone(), secret_value(value)?);
            }
        }
        Ok(())
    }
}

fn secret_value(secret: &SecretString) -> Result<http::HeaderValue> {
    let mut value = header_value(secret.expose())?;
    value.set_sensitive(true);
    Ok(value)
}

/// Async authentication hook, applied to every outgoing request.
///
/// This is the extension seam for refreshable tokens (e.g. OAuth) and request
/// signers (e.g. AWS SigV4): implement it to fetch/refresh a credential and set
/// the appropriate headers. When a provider is configured with an
/// authenticator, it replaces the static [`Credentials`] header logic.
/// Authentication runs last, after profile, provider, and per-request headers
/// and after the body is final, so it has final authority over credential
/// headers and can sign the complete request.
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
    /// Called only for an HTTP 401 or 403 `status` received before any
    /// response output. The request contains the rejected credential headers.
    /// Return `true` only after replacing them and when retrying the request
    /// once is safe.
    ///
    /// # Errors
    ///
    /// Returns an error when recovery or credential refresh fails.
    async fn reauthenticate(&self, _request: &mut HttpRequest, _status: u16) -> Result<bool> {
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
