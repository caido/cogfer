//! AWS request signing for Amazon Bedrock.
//!
//! [`SigV4Authenticator`] is a [`RequestAuthenticator`] that signs each
//! request with AWS Signature Version 4 from static credentials. Hosts that
//! obtain credentials elsewhere (STS, an instance role, SSO) implement their
//! own authenticator and call [`sign_request`] with the credentials they hold.

mod sigv4;

use std::fmt;

pub use self::sigv4::sign_request;
use crate::auth::{Rejection, RequestAuthenticator, SecretString};
use crate::error::Result;
use crate::transport::{HeaderMap, HeaderName, HttpRequest};

/// An AWS access key pair, with the session token of temporary credentials.
#[derive(Clone, PartialEq, Eq)]
pub struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: SecretString,
    pub session_token: Option<SecretString>,
}

impl AwsCredentials {
    pub fn new(
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<SecretString>,
    ) -> Self {
        Self {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token: None,
        }
    }

    /// Temporary credentials carry a session token sent as `x-amz-security-token`.
    #[must_use = "credential modifiers return an updated value"]
    pub fn with_session_token(mut self, token: impl Into<SecretString>) -> Self {
        self.session_token = Some(token.into());
        self
    }
}

impl fmt::Debug for AwsCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AwsCredentials")
            .field("access_key_id", &self.access_key_id)
            .field("session_token", &self.session_token.is_some())
            .finish_non_exhaustive()
    }
}

/// Signs Bedrock requests with static credentials. A 403 naming a signature
/// or clock problem is answered by signing again; other rejections are final.
#[derive(Debug, Clone)]
#[must_use = "authenticators must be attached to a ProviderConfig"]
pub struct SigV4Authenticator {
    credentials: AwsCredentials,
    region: String,
}

impl SigV4Authenticator {
    pub fn new(region: impl Into<String>, credentials: AwsCredentials) -> Self {
        Self {
            credentials,
            region: region.into(),
        }
    }
}

const SERVICE: &str = "bedrock";

#[async_trait::async_trait]
impl RequestAuthenticator for SigV4Authenticator {
    async fn authenticate(&self, request: &mut HttpRequest) -> Result<()> {
        sign_request(request, &self.credentials, &self.region, SERVICE)
    }

    async fn reauthenticate(
        &self,
        request: &mut HttpRequest,
        rejection: &Rejection<'_>,
    ) -> Result<bool> {
        if rejection.status != 403 || !signature_rejected(rejection.headers) {
            return Ok(false);
        }
        self.authenticate(request).await?;
        Ok(true)
    }
}

/// Whether AWS blamed the signature rather than the caller's permissions.
fn signature_rejected(headers: &HeaderMap) -> bool {
    let Some(error_type) = headers
        .get(ERROR_TYPE)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let error_type = error_type.to_ascii_lowercase();
    [
        "signature",
        "requesttimetooskewed",
        "requestexpired",
        "expiredtoken",
    ]
    .iter()
    .any(|cause| error_type.contains(cause))
}

/// The header carrying the AWS exception class on error responses.
const ERROR_TYPE: HeaderName = HeaderName::from_static("x-amzn-errortype");
