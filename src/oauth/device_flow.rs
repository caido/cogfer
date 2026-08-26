//! Shared support for the provider `device_flow` modules: form posts to the
//! auth server, token-response decoding, auth-server error classification, and
//! the RFC 8628 device-code polling loop.

#[cfg(feature = "reqwest-transport")]
use std::future::Future;
use std::time::{Duration, Instant};

use bytes::Bytes;
use serde::Deserialize;
use serde_json::Value;
use url::Url;

#[cfg(feature = "reqwest-transport")]
use super::DevicePoll;
use super::refresh::DEAD_REFRESH_TOKEN_CODES;
use crate::error::{Error, ErrorKind, Result};
use crate::transport::{HeaderMap, HeaderValue, header};
use crate::transport::{HttpRequest, HttpResponse, HttpTransport};

pub(crate) fn require_response_field(
    provider: &str,
    context: &str,
    field: &str,
    value: &str,
) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::malformed(format!(
            "{provider} {context}: {field} must not be empty"
        )));
    }
    Ok(())
}

pub(crate) fn decode_auth_json<T: serde::de::DeserializeOwned>(
    provider: &str,
    context: &str,
    response: &HttpResponse,
) -> Result<T> {
    serde_json::from_slice(&response.body).map_err(|error| {
        Error::malformed(format!(
            "{provider} {context}: invalid response JSON: {error}"
        ))
    })
}

fn device_code_timeout(provider: &'static str) -> Error {
    Error::new(
        ErrorKind::Timeout,
        format!("timed out waiting for the {provider} device code to be confirmed"),
    )
    .with_origin(provider)
}

pub(crate) fn ensure_device_code_is_valid(
    provider: &'static str,
    expires_at: Instant,
) -> Result<()> {
    if Instant::now() >= expires_at {
        return Err(device_code_timeout(provider));
    }
    Ok(())
}

/// POST a form-encoded body and return the raw response for the caller to classify.
pub(crate) async fn post_form(
    transport: &dyn HttpTransport,
    url: Url,
    pairs: &[(&str, &str)],
) -> Result<HttpResponse> {
    let body = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(pairs)
        .finish();
    let request = HttpRequest {
        url,
        headers: HeaderMap::from_iter([
            (header::ACCEPT, HeaderValue::from_static("application/json")),
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/x-www-form-urlencoded"),
            ),
        ]),
        body: Some(Bytes::from(body)),
    };
    transport.execute(request).await
}

/// Poll a device authorization until it completes, sleeping between polls
/// per RFC 8628: the server's `slow_down` and transport timeouts lengthen the
/// interval, and the device code's expiry bounds the whole wait.
#[cfg(feature = "reqwest-transport")]
pub(crate) async fn wait_for_device_tokens<T, F, Fut>(
    provider: &'static str,
    expires_at: Instant,
    poll_interval: Duration,
    mut poll: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<DevicePoll<T>>>,
{
    let deadline = tokio::time::Instant::from_std(expires_at);
    let mut interval = poll_interval;
    loop {
        let remaining = expires_at.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(device_code_timeout(provider));
        }
        tokio::time::sleep(interval.min(remaining)).await;

        let Ok(polled) = tokio::time::timeout_at(deadline, poll()).await else {
            return Err(device_code_timeout(provider));
        };
        match polled {
            Ok(DevicePoll::Complete(tokens)) => return Ok(tokens),
            Ok(DevicePoll::Pending) => {}
            Ok(DevicePoll::SlowDown {
                interval: server_minimum,
            }) => {
                // RFC 8628 §3.5: add five seconds, never below the server minimum.
                let increased = interval.saturating_add(Duration::from_secs(5));
                interval = server_minimum.map_or(increased, |minimum| increased.max(minimum));
            }
            // A transport timeout has no origin, unlike provider timeouts.
            Err(error) if error.kind() == ErrorKind::Timeout && error.origin().is_none() => {
                interval = interval.saturating_mul(2);
            }
            Err(error) => return Err(error),
        }
    }
}

#[derive(Deserialize)]
struct OAuthErrorBody {
    #[serde(default)]
    error: Option<Value>,
    #[serde(default)]
    error_description: Option<String>,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    interval: Option<u64>,
}

pub(crate) struct OAuthErrorParts {
    pub(crate) code: Option<String>,
    pub(crate) detail: String,
    pub(crate) interval: Option<Duration>,
}

pub(crate) fn oauth_error_parts(response: &HttpResponse) -> OAuthErrorParts {
    let parsed: Option<OAuthErrorBody> = serde_json::from_slice(&response.body).ok();
    let (nested_code, nested_message) = match parsed.as_ref().and_then(|body| body.error.as_ref()) {
        Some(Value::String(code)) => (Some(code.clone()), None),
        Some(Value::Object(map)) => (
            map.get("code").and_then(Value::as_str).map(str::to_owned),
            map.get("message")
                .and_then(Value::as_str)
                .map(str::to_owned),
        ),
        _ => (None, None),
    };
    let code = nested_code.or_else(|| parsed.as_ref().and_then(|body| body.code.clone()));
    let detail = parsed
        .as_ref()
        .and_then(|body| body.error_description.clone())
        .or(nested_message)
        .or_else(|| parsed.as_ref().and_then(|body| body.message.clone()))
        .filter(|description| !description.trim().is_empty())
        .or_else(|| code.clone())
        .unwrap_or_else(|| {
            crate::util::truncate_for_error(&String::from_utf8_lossy(&response.body), 200)
        });
    let interval = parsed
        .as_ref()
        .and_then(|body| body.interval)
        .map(Duration::from_secs);
    OAuthErrorParts {
        code,
        detail,
        interval,
    }
}

pub(crate) fn auth_error(provider: &'static str, context: &str, response: &HttpResponse) -> Error {
    let parts = oauth_error_parts(response);
    let mut error = Error::new(
        crate::http::error_kind_for_status(response.status),
        format!("{provider} {context} failed: {}", parts.detail),
    )
    .with_origin(provider)
    .with_status(response.status);
    if let Some(code) = parts.code {
        error = error.with_code(code);
    }
    error
}

pub(crate) fn refresh_error(provider: &'static str, response: &HttpResponse) -> Error {
    let parts = oauth_error_parts(response);
    if matches!(response.status, 400 | 401 | 403)
        && let Some(code) = parts
            .code
            .as_deref()
            .filter(|code| DEAD_REFRESH_TOKEN_CODES.contains(code))
    {
        return Error::new(
            ErrorKind::Authentication,
            format!(
                "{provider} refresh token is no longer valid ({}); sign in again",
                parts.detail
            ),
        )
        .with_origin(provider)
        .with_status(response.status)
        .with_code(code);
    }
    let mut error = Error::new(
        crate::http::error_kind_for_status(response.status),
        format!("{provider} token refresh failed: {}", parts.detail),
    )
    .with_origin(provider)
    .with_status(response.status);
    if let Some(code) = parts.code {
        error = error.with_code(code);
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expired_device_codes_time_out() {
        let error = ensure_device_code_is_valid("provider", Instant::now()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Timeout);
        assert!(
            ensure_device_code_is_valid("provider", Instant::now() + Duration::from_secs(60))
                .is_ok()
        );
    }
}
