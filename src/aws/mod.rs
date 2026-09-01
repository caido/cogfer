//! AWS request signing for Amazon Bedrock.
//!
//! [`SigV4Authenticator`] is a [`RequestAuthenticator`] that signs each
//! request with AWS Signature Version 4. It takes any [`ProvideCredentials`]
//! implementation: static [`AwsCredentials`], or a custom provider for
//! credentials that rotate (STS, an instance role, SSO). Hosts with their own
//! signing pipeline can call [`sign_request`] directly.
//!
//! Credential types are re-exported from [`aws_credential_types`], with
//! `Credentials` aliased to [`AwsCredentials`] so it cannot be confused with
//! [`crate::Credentials`].

mod sigv4;

use std::fmt;

pub use aws_credential_types::Credentials as AwsCredentials;
pub use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider, future};

pub use self::sigv4::sign_request;
use crate::auth::{Rejection, RequestAuthenticator};
use crate::error::{Error, ErrorKind, Result};
use crate::protocols::bedrock::{ERROR_TYPE, is_signature_failure};
use crate::transport::{HeaderMap, HttpRequest};

/// Signs Bedrock requests. A 403 naming a signature, clock, or expired-token
/// problem is answered by fetching credentials and signing again. Other
/// rejections are final.
#[derive(Clone)]
#[must_use = "authenticators must be attached to a ProviderConfig"]
pub struct SigV4Authenticator {
    provider: SharedCredentialsProvider,
    region: String,
}

impl fmt::Debug for SigV4Authenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SigV4Authenticator")
            .field("region", &self.region)
            .finish_non_exhaustive()
    }
}

impl SigV4Authenticator {
    /// Sign with what `provider` returns for each request.
    ///
    /// [`AwsCredentials`] are their own provider, so static keys pass
    /// directly. Rotating credentials (an STS session, an instance role)
    /// implement [`ProvideCredentials`], which is called before every
    /// signature so they stay current.
    pub fn new(region: impl Into<String>, provider: impl ProvideCredentials + 'static) -> Self {
        Self {
            provider: SharedCredentialsProvider::new(provider),
            region: region.into(),
        }
    }
}

const SERVICE: &str = "bedrock";

#[async_trait::async_trait]
impl RequestAuthenticator for SigV4Authenticator {
    async fn authenticate(&self, request: &mut HttpRequest) -> Result<()> {
        let credentials = self.provider.provide_credentials().await.map_err(|error| {
            Error::new(
                ErrorKind::Authentication,
                "aws: credentials could not be provided",
            )
            .with_source(error)
        })?;
        sign_request(request, &credentials, &self.region, SERVICE)
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
///
/// The vocabulary is [`is_signature_failure`], shared with the classifier that
/// turns the same exceptions into [`crate::ErrorKind::Authentication`], so a
/// rejection cannot be called an auth failure here and retried differently
/// there.
fn signature_rejected(headers: &HeaderMap) -> bool {
    headers
        .get(ERROR_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(is_signature_failure)
}
