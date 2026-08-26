use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::Value;
use url::Url;

use super::tokens::ChatGptTokens;
use crate::error::{Error, Result};
use crate::oauth::{
    DevicePoll, OAuthClientConfig, auth_error, decode_auth_json, ensure_device_code_is_valid,
    expires_at_from_lifetime, oauth_error_parts, post_form, refresh_error, require_response_field,
};
use crate::transport::{HttpRequest, HttpResponse, HttpTransport};

const PROVIDER: &str = "chatgpt";

/// The user has this long to confirm a device code before
/// [`ChatGptOAuth::wait_for_tokens`] gives up.
const DEVICE_FLOW_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// A pending device authorization to show to the user and then poll.
#[derive(Clone)]
pub struct DeviceAuthorization {
    /// The page where the user enters the code.
    pub verification_url: String,
    /// The code the user enters. Never share it anywhere else.
    pub user_code: String,
    /// The poll cadence the backend asked for.
    pub poll_interval: Duration,
    device_auth_id: String,
    expires_at: Instant,
}

impl fmt::Debug for DeviceAuthorization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceAuthorization")
            .field("verification_url", &self.verification_url)
            .field("user_code", &"<redacted>")
            .field("poll_interval", &self.poll_interval)
            .field("device_auth_id", &"<redacted>")
            .finish()
    }
}

/// ChatGPT device-code sign-in and token refresh client.
///
/// Each operation is one request over the injected [`HttpTransport`].
/// [`ChatGptOAuth::poll_device_authorization`] leaves sleeping to the caller.
/// `wait_for_tokens` is the optional Tokio convenience loop.
#[derive(Clone)]
#[must_use = "OAuth client modifiers return an updated value"]
pub struct ChatGptOAuth {
    transport: Arc<dyn HttpTransport>,
    auth_base_url: Url,
    client_config: OAuthClientConfig,
}

impl fmt::Debug for ChatGptOAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatGptOAuth")
            .field("auth_base_url", &self.auth_base_url.as_str())
            .finish_non_exhaustive()
    }
}

impl ChatGptOAuth {
    /// Create an OAuth client using the standard ChatGPT auth endpoint.
    ///
    /// The default identity is OpenAI Codex's public device-flow client. Hosts
    /// with their own registration should use
    /// [`ChatGptOAuth::with_client_config`] and persist it with the tokens.
    ///
    /// # Panics
    ///
    /// Panics only if the auth endpoint embedded in this crate is not a valid
    /// URL, which indicates a library bug.
    pub fn new(transport: Arc<dyn HttpTransport>) -> Self {
        Self {
            transport,
            auth_base_url: Url::parse("https://auth.openai.com")
                .expect("default auth URL is valid"),
            // Codex's public OAuth client id. Device-flow clients have no secret.
            client_config: OAuthClientConfig::new("app_EMoamEEZ73f0CkXaXp7hrann")
                .expect("default OAuth client id is valid"),
        }
    }

    /// Use the built-in reqwest transport.
    ///
    /// # Errors
    ///
    /// Returns an error if the reqwest transport cannot be initialized, for
    /// example when no Rustls crypto provider is installed (see
    /// [`install_default_crypto_provider`](crate::transport::install_default_crypto_provider)).
    #[cfg(feature = "reqwest-transport")]
    pub fn with_default_transport() -> Result<Self> {
        Ok(Self::new(Arc::new(
            crate::transport::ReqwestTransport::new()?,
        )))
    }

    /// Use a client registration that matches the issued refresh tokens.
    pub fn with_client_config(mut self, config: OAuthClientConfig) -> Self {
        self.client_config = config;
        self
    }

    /// Begin a device-code sign-in: returns the code to show the user.
    ///
    /// # Errors
    ///
    /// Returns an error when the request cannot be built or sent, the auth
    /// server rejects it, or its response is not valid device-code JSON.
    pub async fn start_device_authorization(&self) -> Result<DeviceAuthorization> {
        #[derive(Deserialize)]
        struct DeviceCodeResponse {
            device_auth_id: String,
            #[serde(alias = "usercode")]
            user_code: String,
            #[serde(default)]
            interval: Option<Value>,
        }

        let url = crate::http::join_url(&self.auth_base_url, "api/accounts/deviceauth/usercode");
        let request = HttpRequest::post_json(
            url,
            &serde_json::json!({ "client_id": self.client_config.client_id() }),
        )?;
        let response = self.transport.execute(request).await?;
        let issued_at = Instant::now();
        if !(200..300).contains(&response.status) {
            return Err(auth_error(PROVIDER, "device authorization", &response));
        }
        let parsed: DeviceCodeResponse =
            decode_auth_json(PROVIDER, "device authorization", &response)?;
        require_response_field(
            PROVIDER,
            "device authorization",
            "device_auth_id",
            &parsed.device_auth_id,
        )?;
        require_response_field(
            PROVIDER,
            "device authorization",
            "user_code",
            &parsed.user_code,
        )?;
        let expires_at = issued_at.checked_add(DEVICE_FLOW_TIMEOUT).ok_or_else(|| {
            Error::malformed("chatgpt device authorization lifetime exceeds the platform clock")
        })?;

        let verification_url = crate::http::join_url(&self.auth_base_url, "codex/device").into();
        Ok(DeviceAuthorization {
            verification_url,
            user_code: parsed.user_code,
            // The backend reports the interval as a number or a string.
            poll_interval: parsed
                .interval
                .and_then(|value| match value {
                    Value::Number(seconds) => seconds.as_u64(),
                    Value::String(seconds) => seconds.trim().parse().ok(),
                    _ => None,
                })
                .map(|seconds| Duration::from_secs(seconds.max(1)))
                .unwrap_or(Duration::from_secs(5)),
            device_auth_id: parsed.device_auth_id,
            expires_at,
        })
    }

    /// One poll of a pending device authorization.
    ///
    /// # Errors
    ///
    /// Returns an error when polling or exchanging the authorization code
    /// fails, or when the auth server returns a terminal or malformed response.
    pub async fn poll_device_authorization(
        &self,
        device: &DeviceAuthorization,
    ) -> Result<DevicePoll<ChatGptTokens>> {
        ensure_device_code_is_valid(PROVIDER, device.expires_at)?;

        #[derive(Deserialize)]
        struct DeviceTokenResponse {
            authorization_code: String,
            code_verifier: String,
        }

        let url = crate::http::join_url(&self.auth_base_url, "api/accounts/deviceauth/token");
        let request = HttpRequest::post_json(
            url,
            &serde_json::json!({
                "device_auth_id": device.device_auth_id,
                "user_code": device.user_code,
            }),
        )?;
        let response = self.transport.execute(request).await?;
        // The backend answers a pending code with 403/404 rather than 400.
        if matches!(response.status, 403 | 404) {
            let parts = oauth_error_parts(&response);
            return match parts.code.as_deref() {
                None => Ok(DevicePoll::Pending),
                Some(code) if code.contains("pending") => Ok(DevicePoll::Pending),
                Some("slow_down") => Ok(DevicePoll::SlowDown {
                    interval: parts.interval,
                }),
                Some(_) => Err(auth_error(PROVIDER, "device authorization", &response)),
            };
        }
        if !(200..300).contains(&response.status) {
            return Err(auth_error(PROVIDER, "device authorization", &response));
        }
        let parsed: DeviceTokenResponse =
            decode_auth_json(PROVIDER, "device authorization", &response)?;
        require_response_field(
            PROVIDER,
            "device authorization",
            "authorization_code",
            &parsed.authorization_code,
        )?;
        require_response_field(
            PROVIDER,
            "device authorization",
            "code_verifier",
            &parsed.code_verifier,
        )?;
        self.exchange_authorization_code(&parsed.authorization_code, &parsed.code_verifier)
            .await
            .map(DevicePoll::Complete)
    }

    /// Poll until the user confirms the device code, sleeping between polls.
    /// Gives up with [`crate::ErrorKind::Timeout`] 15 minutes after the code is issued.
    ///
    /// # Errors
    ///
    /// Returns an error when polling or token exchange fails or the device
    /// code times out.
    #[cfg(feature = "reqwest-transport")]
    pub async fn wait_for_tokens(&self, device: &DeviceAuthorization) -> Result<ChatGptTokens> {
        crate::oauth::wait_for_device_tokens(
            PROVIDER,
            device.expires_at,
            device.poll_interval,
            || self.poll_device_authorization(device),
        )
        .await
    }

    /// Exchange a refresh token for a fresh token set. The previous refresh
    /// token is kept when the server does not rotate it.
    ///
    /// # Errors
    ///
    /// Returns an error when the refresh request fails, the auth server rejects
    /// the token, or its response is malformed.
    pub async fn refresh(&self, refresh_token: &str) -> Result<ChatGptTokens> {
        if refresh_token.trim().is_empty() {
            return Err(Error::invalid_request(
                "ChatGPT refresh token must not be empty",
            ));
        }
        let response = self
            .post_token_form(&[
                ("client_id", self.client_config.client_id()),
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token),
            ])
            .await?;

        if !(200..300).contains(&response.status) {
            return Err(refresh_error(PROVIDER, &response));
        }
        let mut tokens = decode_token_response("token refresh", &response)?;
        if tokens.refresh_token.is_none() {
            tokens.refresh_token = Some(refresh_token.to_string());
        }
        Ok(tokens)
    }

    async fn exchange_authorization_code(
        &self,
        authorization_code: &str,
        code_verifier: &str,
    ) -> Result<ChatGptTokens> {
        let redirect_uri =
            crate::http::join_url(&self.auth_base_url, "deviceauth/callback").to_string();
        let response = self
            .post_token_form(&[
                ("grant_type", "authorization_code"),
                ("code", authorization_code),
                ("redirect_uri", &redirect_uri),
                ("client_id", self.client_config.client_id()),
                ("code_verifier", code_verifier),
            ])
            .await?;

        if !(200..300).contains(&response.status) {
            return Err(auth_error(PROVIDER, "token exchange", &response));
        }
        decode_token_response("token exchange", &response)
    }

    async fn post_token_form(&self, pairs: &[(&str, &str)]) -> Result<HttpResponse> {
        let url = crate::http::join_url(&self.auth_base_url, "oauth/token");
        post_form(self.transport.as_ref(), url, pairs).await
    }
}

fn decode_token_response(context: &str, response: &HttpResponse) -> Result<ChatGptTokens> {
    #[derive(Deserialize)]
    struct TokenResponse {
        access_token: String,
        #[serde(default)]
        refresh_token: Option<String>,
        #[serde(default)]
        id_token: Option<String>,
        #[serde(default)]
        expires_in: Option<i64>,
    }

    let parsed: TokenResponse = decode_auth_json(PROVIDER, context, response)?;
    require_response_field(PROVIDER, context, "access_token", &parsed.access_token)?;
    if let Some(refresh_token) = &parsed.refresh_token {
        require_response_field(PROVIDER, context, "refresh_token", refresh_token)?;
    }
    if let Some(id_token) = &parsed.id_token {
        require_response_field(PROVIDER, context, "id_token", id_token)?;
    }
    let expires_at = parsed
        .expires_in
        .map(|seconds| expires_at_from_lifetime(PROVIDER, context, seconds))
        .transpose()?;
    // The JWT `exp` claim is authoritative. `expires_in` covers opaque tokens.
    let mut tokens = ChatGptTokens {
        access_token: parsed.access_token,
        refresh_token: parsed.refresh_token,
        id_token: parsed.id_token,
        expires_at: None,
        account_id: None,
    }
    .derive_metadata();
    if tokens.expires_at.is_none() {
        tokens.expires_at = expires_at;
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;
    use crate::transport::HeaderMap;

    #[test]
    fn debug_output_redacts_authorization_codes() {
        let authorization = DeviceAuthorization {
            verification_url: "https://example.com/device".into(),
            user_code: "secret-user-code".into(),
            poll_interval: Duration::from_secs(5),
            device_auth_id: "secret-device-auth-id".into(),
            expires_at: Instant::now() + DEVICE_FLOW_TIMEOUT,
        };

        let debug = format!("{authorization:?}");
        assert!(!debug.contains("secret-user-code"));
        assert!(!debug.contains("secret-device-auth-id"));
    }

    #[test]
    fn token_response_rejects_empty_access_token() {
        let response = HttpResponse {
            status: 200,
            headers: HeaderMap::new(),
            body: Bytes::from_static(br#"{"access_token":""}"#),
        };

        let error = decode_token_response("token refresh", &response)
            .expect_err("empty access token should be rejected");

        assert_eq!(error.kind(), crate::ErrorKind::MalformedResponse);
    }
}
