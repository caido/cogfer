//! The [`LanguageModel`] execution handle.

use crate::capabilities::ModelCapabilities;
use crate::error::Result;
use crate::provider::Provider;
use crate::request::Request;
use crate::response::GenerateResult;
use crate::stream::EventStream;

/// A handle to one model on one provider. Cheap to clone.
#[derive(Debug, Clone)]
pub struct LanguageModel {
    pub(crate) provider: Provider,
    pub(crate) id: String,
    pub(crate) capabilities: ModelCapabilities,
}

impl LanguageModel {
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The settings this model accepts. Requests are fitted to them before
    /// lowering, so hosts can use this to hide the controls that would be
    /// dropped.
    pub fn capabilities(&self) -> &ModelCapabilities {
        &self.capabilities
    }

    /// Narrow the capabilities to what `model` says this model supports, for
    /// example from a models.dev entry. The profile defaults remain the
    /// upper bound: nothing the wire format cannot carry is enabled.
    #[must_use = "model modifiers return an updated value"]
    pub fn with_capabilities(mut self, model: &ModelCapabilities) -> Self {
        self.capabilities = self.capabilities.restrict(model);
        self
    }

    /// Execute a non-streaming completion.
    ///
    /// # Errors
    ///
    /// Returns an error when the request cannot be lowered, the transport
    /// fails, or the provider returns an error or malformed response.
    pub async fn generate(&self, request: Request) -> Result<GenerateResult> {
        crate::protocols::runner::generate(&self.provider, &self.id, &self.capabilities, request)
            .await
    }

    /// Execute a streaming completion. See [`EventStream`] for the terminal
    /// contract. Failures before the stream is established return `Err` here.
    ///
    /// # Errors
    ///
    /// Returns an error when the request cannot be lowered, the transport
    /// fails before streaming begins, or the initial provider response is invalid.
    pub async fn stream(&self, request: Request) -> Result<EventStream> {
        crate::protocols::runner::stream(&self.provider, &self.id, &self.capabilities, request)
            .await
    }
}
