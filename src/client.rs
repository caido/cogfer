//! The SDK entry point.

use std::fmt;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig, ProviderInner};
use crate::transport::HttpTransport;

/// The SDK client: a transport plus provider construction.
#[derive(Clone)]
pub struct Client {
    transport: Arc<dyn HttpTransport>,
}

impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client").finish_non_exhaustive()
    }
}

impl Client {
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    /// Create a provider connection from configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured base URL cannot be used for HTTP requests.
    pub fn provider(&self, config: ProviderConfig) -> Result<Provider> {
        if let Some(url) = config.custom_base_url() {
            if !matches!(url.scheme(), "http" | "https") {
                return Err(Error::configuration(format!(
                    "unsupported base URL scheme `{}`",
                    url.scheme()
                )));
            }
            if url.cannot_be_a_base() {
                return Err(Error::configuration("base URL cannot be a base"));
            }
        }
        Ok(Provider {
            inner: Arc::new(ProviderInner {
                config,
                transport: self.transport.clone(),
            }),
        })
    }
}

/// Builder for [`Client`].
#[derive(Default)]
#[must_use = "client builders do nothing until build is called"]
pub struct ClientBuilder {
    transport: Option<Arc<dyn HttpTransport>>,
}

impl fmt::Debug for ClientBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClientBuilder").finish_non_exhaustive()
    }
}

impl ClientBuilder {
    /// Inject a custom transport (any [`HttpTransport`] implementation), for
    /// example the built-in reqwest transport wrapping a preconfigured
    /// reqwest client.
    pub fn http_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.transport = Some(transport);
        self
    }

    /// Build the configured SDK client.
    ///
    #[cfg_attr(
        feature = "reqwest-transport",
        doc = "Without an injected transport this builds a \
               [`ReqwestTransport`](crate::transport::ReqwestTransport), which needs a \
               process-wide Rustls crypto provider; see \
               [`install_default_crypto_provider`](crate::transport::install_default_crypto_provider)."
    )]
    ///
    /// # Errors
    ///
    /// Returns an error when the default HTTP client cannot be constructed,
    /// or when no transport was provided and the built-in transport is disabled.
    pub fn build(self) -> Result<Client> {
        #[cfg(feature = "reqwest-transport")]
        {
            let transport: Arc<dyn HttpTransport> = match self.transport {
                Some(transport) => transport,
                None => Arc::new(crate::transport::ReqwestTransport::new()?),
            };
            Ok(Client { transport })
        }
        #[cfg(not(feature = "reqwest-transport"))]
        {
            let transport = self.transport.ok_or_else(|| {
                Error::configuration(
                    "no transport configured: enable the `reqwest-transport` feature or inject one with `http_transport`",
                )
            })?;
            Ok(Client { transport })
        }
    }
}
