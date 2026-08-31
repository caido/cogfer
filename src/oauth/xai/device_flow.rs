use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use url::Url;

use super::tokens::XaiTokens;
use crate::error::{Error, ErrorKind, Result};
use crate::oauth::{
    DevicePoll, OAuthClientConfig, auth_error, decode_auth_json, ensure_device_code_is_valid,
    expires_at_from_lifetime, oauth_error_parts, post_form, refresh_error, require_response_field,
};
use crate::transport::{HttpResponse, HttpTransport};

const PROVIDER: &str = "xai";

/// A pending device-code authorization.
#[derive(Clone)]
pub struct DeviceAuthorization {
    /// The page where the user enters the code.
    pub verification_url: String,
    /// Verification URL with the code pre-filled when available.
    pub verification_url_complete: Option<String>,
    /// The code the user enters. Never share it anywhere else.
    pub user_code: String,
    /// The poll cadence the server asked for.
    pub poll_interval: Duration,
    /// How long the device code stays valid.
    pub expires_in: Duration,
    device_code: String,
    expires_at: Instant,
}

impl fmt::Debug for DeviceAuthorization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceAuthorization")
            .field("verification_url", &self.verification_url)
            .field(
                "verification_url_complete",
                &self
                    .verification_url_complete
                    .as_ref()
                    .map(|_| "<redacted>"),
            )
            .field("user_code", &"<redacted>")
            .field("poll_interval", &self.poll_interval)
            .field("expires_in", &self.expires_in)
            .field("device_code", &"<redacted>")
            .finish()
    }
}

/// SpaceXAI device-code sign-in and token refresh client.
///
/// Each operation is one request over the injected [`HttpTransport`].
/// [`XaiOAuth::poll_device_authorization`] leaves sleeping to the caller.
/// `wait_for_tokens` is the optional Tokio convenience loop.
#[derive(Clone)]
#[must_use = "OAuth client modifiers return an updated value"]
pub struct XaiOAuth {
    transport: Arc<dyn HttpTransport>,
    auth_base_url: Url,
    client_config: OAuthClientConfig,
}

impl fmt::Debug for XaiOAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XaiOAuth")
            .field("auth_base_url", &self.auth_base_url.as_str())
            .finish_non_exhaustive()
    }
}

impl XaiOAuth {
    /// Create an OAuth client using the standard SpaceXAI auth endpoint.
    ///
    /// The defaults are grok-cli's public identity and scopes. Hosts with
    /// their own registration should use [`XaiOAuth::with_client_config`] and
    /// persist it with the tokens.
    ///
    /// # Panics
    ///
    /// Panics only if the auth endpoint embedded in this crate is not a valid
    /// URL, which indicates a library bug.
    pub fn new(transport: Arc<dyn HttpTransport>) -> Self {
        Self {
            transport,
            auth_base_url: Url::parse("https://auth.x.ai").expect("default auth URL is valid"),
            // grok-cli's public client id (SpaceXAI binds subscription
            // tokens to it) and the scopes it needs to reach `api.x.ai`.
            client_config: OAuthClientConfig::scoped(
                "b1a00492-073a-47ea-816f-4c329264a828",
                "openid profile email offline_access grok-cli:access api:access",
            )
            .expect("default OAuth client configuration is valid"),
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

    /// Send device authorization and token requests to another auth server,
    /// for example a local stand-in during tests.
    pub fn with_auth_base_url(mut self, auth_base_url: Url) -> Self {
        self.auth_base_url = auth_base_url;
        self
    }

    /// Begin a device-code sign-in: returns the code to show the user.
    ///
    /// # Errors
    ///
    /// Returns an error when the request fails, the auth server rejects it,
    /// its response is malformed, a verification URL is not HTTPS, or the
    /// advertised lifetime cannot be represented by the platform clock.
    pub async fn start_device_authorization(&self) -> Result<DeviceAuthorization> {
        #[derive(Deserialize)]
        struct DeviceCodeResponse {
            device_code: String,
            user_code: String,
            verification_uri: String,
            #[serde(default)]
            verification_uri_complete: Option<String>,
            #[serde(default)]
            interval: Option<u64>,
            expires_in: u64,
        }

        let mut form = vec![("client_id", self.client_config.client_id())];
        if let Some(scope) = self.client_config.scope() {
            form.push(("scope", scope));
        }
        form.push(("referrer", "llmwire"));
        let response = self.post_form("oauth2/device/code", &form).await?;
        let issued_at = Instant::now();
        if !(200..300).contains(&response.status) {
            return Err(auth_error(PROVIDER, "device authorization", &response));
        }
        let parsed: DeviceCodeResponse =
            decode_auth_json(PROVIDER, "device authorization", &response)?;
        require_response_field(
            PROVIDER,
            "device authorization",
            "device_code",
            &parsed.device_code,
        )?;
        require_response_field(
            PROVIDER,
            "device authorization",
            "user_code",
            &parsed.user_code,
        )?;
        let expires_in = Duration::from_secs(parsed.expires_in);
        let expires_at = device_code_expiry(issued_at, expires_in)?;

        Ok(DeviceAuthorization {
            verification_url: require_https("verification_uri", parsed.verification_uri)?,
            verification_url_complete: parsed
                .verification_uri_complete
                .map(|url| require_https("verification_uri_complete", url))
                .transpose()?,
            user_code: parsed.user_code,
            // RFC 8628 §3.2: poll every five seconds unless the server says otherwise.
            poll_interval: parsed
                .interval
                .map(|seconds| Duration::from_secs(seconds.max(1)))
                .unwrap_or(Duration::from_secs(5)),
            expires_in,
            device_code: parsed.device_code,
            expires_at,
        })
    }

    /// Poll a pending authorization once.
    ///
    /// # Errors
    ///
    /// Returns an error when the request fails, the server rejects or expires
    /// the device code, or a successful token response is malformed.
    pub async fn poll_device_authorization(
        &self,
        device: &DeviceAuthorization,
    ) -> Result<DevicePoll<XaiTokens>> {
        ensure_device_code_is_valid(PROVIDER, device.expires_at)?;
        let response = self
            .post_form(
                "oauth2/token",
                &[
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ("client_id", self.client_config.client_id()),
                    ("device_code", &device.device_code),
                ],
            )
            .await?;

        if (200..300).contains(&response.status) {
            let tokens = decode_token_response("device token", &response, None)?;
            return Ok(DevicePoll::Complete(tokens));
        }

        let parts = oauth_error_parts(&response);
        match parts.code.as_deref() {
            Some("authorization_pending") => Ok(DevicePoll::Pending),
            Some("slow_down") => Ok(DevicePoll::SlowDown {
                interval: parts.interval,
            }),
            Some("access_denied" | "authorization_denied") => Err(Error::new(
                ErrorKind::Authentication,
                format!("xai device authorization was denied: {}", parts.detail),
            )
            .with_origin(PROVIDER)
            .with_status(response.status)),
            Some("expired_token") => Err(Error::new(
                ErrorKind::Timeout,
                "xai device code expired before it was confirmed; restart the sign-in",
            )
            .with_origin(PROVIDER)
            .with_status(response.status)),
            _ => Err(auth_error(PROVIDER, "device token polling", &response)),
        }
    }

    /// Poll until the user confirms the device code, sleeping between polls
    /// and honouring `slow_down` answers. Gives up with
    /// [`ErrorKind::Timeout`] when the device code expires.
    ///
    /// # Errors
    ///
    /// Returns an error when polling fails or the device code expires or is denied.
    #[cfg(feature = "reqwest-transport")]
    pub async fn wait_for_tokens(&self, device: &DeviceAuthorization) -> Result<XaiTokens> {
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
    pub async fn refresh(&self, refresh_token: &str) -> Result<XaiTokens> {
        if refresh_token.trim().is_empty() {
            return Err(Error::invalid_request(
                "SpaceXAI refresh token must not be empty",
            ));
        }
        let response = self
            .post_form(
                "oauth2/token",
                &[
                    ("grant_type", "refresh_token"),
                    ("client_id", self.client_config.client_id()),
                    ("refresh_token", refresh_token),
                ],
            )
            .await?;
        if !(200..300).contains(&response.status) {
            return Err(refresh_error(PROVIDER, &response));
        }
        decode_token_response("token refresh", &response, Some(refresh_token))
    }

    async fn post_form(&self, path: &str, pairs: &[(&str, &str)]) -> Result<HttpResponse> {
        let url = crate::http::join_url(&self.auth_base_url, path);
        post_form(self.transport.as_ref(), url, pairs).await
    }
}

/// Decode a token response. `previous_refresh_token` is kept when the
/// server does not rotate it (SpaceXAI omits `refresh_token` then).
fn decode_token_response(
    context: &str,
    response: &HttpResponse,
    previous_refresh_token: Option<&str>,
) -> Result<XaiTokens> {
    #[derive(Deserialize)]
    struct TokenResponse {
        access_token: String,
        #[serde(default)]
        refresh_token: Option<String>,
        #[serde(default)]
        expires_in: Option<i64>,
    }

    let parsed: TokenResponse = decode_auth_json(PROVIDER, context, response)?;
    require_response_field(PROVIDER, context, "access_token", &parsed.access_token)?;
    if let Some(refresh_token) = &parsed.refresh_token {
        require_response_field(PROVIDER, context, "refresh_token", refresh_token)?;
    }
    // SpaceXAI omits `expires_in` and its tokens live about an hour.
    let expires_at =
        expires_at_from_lifetime(PROVIDER, context, parsed.expires_in.unwrap_or(3600))?;
    Ok(XaiTokens {
        access_token: parsed.access_token,
        refresh_token: parsed
            .refresh_token
            .or_else(|| previous_refresh_token.map(str::to_owned)),
        expires_at: Some(expires_at),
    })
}

fn device_code_expiry(issued_at: Instant, expires_in: Duration) -> Result<Instant> {
    if expires_in.is_zero() {
        return Err(Error::malformed(
            "xai device authorization: expires_in must be greater than zero",
        ));
    }
    issued_at.checked_add(expires_in).ok_or_else(|| {
        Error::malformed("xai device authorization: expires_in exceeds the platform clock range")
    })
}

/// Reject verification URLs that could launch a non-web handler.
fn require_https(field: &str, raw: String) -> Result<String> {
    match Url::parse(&raw) {
        Ok(url) if url.scheme() == "https" => Ok(raw),
        _ => Err(Error::malformed(format!(
            "xai device authorization: {field} is not an https URL"
        ))),
    }
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
            verification_url_complete: Some(
                "https://example.com/device?code=secret-user-code".into(),
            ),
            user_code: "secret-user-code".into(),
            poll_interval: Duration::from_secs(5),
            expires_in: Duration::from_secs(900),
            device_code: "secret-device-code".into(),
            expires_at: Instant::now() + Duration::from_secs(900),
        };

        let debug = format!("{authorization:?}");
        assert!(!debug.contains("secret-user-code"));
        assert!(!debug.contains("secret-device-code"));
    }

    #[test]
    fn device_code_lifetime_rejects_platform_clock_overflow() {
        let error = device_code_expiry(Instant::now(), Duration::MAX)
            .expect_err("unrepresentable device-code expiry should be rejected");

        assert_eq!(error.kind(), ErrorKind::MalformedResponse);
    }

    #[test]
    fn https_is_required_on_verification_urls() {
        assert!(require_https("verification_uri", "https://x.ai/device".into()).is_ok());
        assert!(require_https("verification_uri", "http://x.ai/device".into()).is_err());
        assert!(require_https("verification_uri", "file:///etc/passwd".into()).is_err());
        assert!(require_https("verification_uri", "not a url".into()).is_err());
    }

    #[test]
    fn missing_expires_in_defaults_to_an_hour() {
        let response = HttpResponse {
            status: 200,
            headers: HeaderMap::new(),
            body: Bytes::from_static(br#"{"access_token":"opaque"}"#),
        };

        let tokens = decode_token_response("token refresh", &response, Some("keep")).unwrap();

        assert_eq!(tokens.refresh_token.as_deref(), Some("keep"));
        assert!(
            tokens
                .expires_at
                .is_some_and(|at| at > crate::oauth::unix_now() + 3500)
        );
    }
}
