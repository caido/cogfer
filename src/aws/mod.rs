//! AWS request signing for Amazon Bedrock.
//!
//! [`SigV4Authenticator`] is a [`RequestAuthenticator`] that signs each
//! request with AWS Signature Version 4 from static credentials. Hosts that
//! obtain credentials elsewhere (STS, an instance role, SSO) implement their
//! own authenticator and call [`sign_request`] with the credentials they hold.

mod sigv4;

use std::fmt;
use std::time::SystemTime;

pub use self::sigv4::sign_request;
use crate::auth::{RequestAuthenticator, SecretString};
use crate::error::Result;
use crate::transport::HttpRequest;

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

/// Signs requests with static credentials. Bedrock rejects a signature
/// whose timestamp has drifted, so a 403 is answered by signing again.
#[derive(Debug, Clone)]
#[must_use = "authenticators must be attached to a ProviderConfig"]
pub struct SigV4Authenticator {
    credentials: AwsCredentials,
    region: String,
    service: String,
}

impl SigV4Authenticator {
    /// Sign for the `bedrock` service in `region`.
    pub fn new(region: impl Into<String>, credentials: AwsCredentials) -> Self {
        Self {
            credentials,
            region: region.into(),
            service: "bedrock".into(),
        }
    }

    /// Sign for another AWS service.
    pub fn with_service(mut self, service: impl Into<String>) -> Self {
        self.service = service.into();
        self
    }
}

#[async_trait::async_trait]
impl RequestAuthenticator for SigV4Authenticator {
    async fn authenticate(&self, request: &mut HttpRequest) -> Result<()> {
        sign_request(
            request,
            &self.credentials,
            &self.region,
            &self.service,
            SystemTime::now(),
        )
    }

    async fn reauthenticate(&self, request: &mut HttpRequest, status: u16) -> Result<bool> {
        if status != 403 {
            return Ok(false);
        }
        self.authenticate(request).await?;
        Ok(true)
    }
}
