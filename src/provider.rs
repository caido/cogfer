//! Provider configuration and the [`Provider`] handle.

use std::fmt;
use std::sync::Arc;

use url::Url;

use crate::auth::{Credentials, RequestAuthenticator};
use crate::capabilities::ModelCapabilities;
use crate::error::{Error, Result};
use crate::model::LanguageModel;
use crate::protocols::ApiProfile;
use crate::transport::{HeaderMap, HeaderName, HeaderValue, HttpTransport};

#[derive(Clone)]
pub(crate) enum Authentication {
    Credentials(Credentials),
    Authenticator(Arc<dyn RequestAuthenticator>),
}

/// Configuration for one provider connection.
///
/// Request preparation applies profile-required headers first, then provider
/// headers, then [`crate::Request::extra_headers`], and finally authentication.
/// Later values replace earlier values case-insensitively, except
/// `anthropic-beta`, whose comma-separated feature values are merged.
#[derive(Clone)]
#[must_use = "provider configuration must be passed to Client::provider"]
pub struct ProviderConfig {
    profile: ApiProfile,
    authentication: Authentication,
    /// Custom base URL whose path prefix is preserved. `None` uses the official endpoint.
    base_url: Option<Url>,
    /// Headers added to every request from this provider.
    default_headers: HeaderMap,
}

impl fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let base_url = self.base_url.as_ref().map(crate::http::sanitized_url);
        f.debug_struct("ProviderConfig")
            .field("profile", &self.profile)
            .field("base_url", &base_url)
            .field("default_headers", &self.default_headers.len())
            .field(
                "authenticator",
                &matches!(&self.authentication, Authentication::Authenticator(_)),
            )
            .finish()
    }
}

impl ProviderConfig {
    /// Configure an API profile with static credentials.
    pub fn new(profile: ApiProfile, credentials: Credentials) -> Self {
        Self {
            profile,
            authentication: Authentication::Credentials(credentials),
            base_url: None,
            default_headers: HeaderMap::new(),
        }
    }

    /// OpenAI Responses API.
    pub fn openai_responses(credentials: Credentials) -> Self {
        Self::new(ApiProfile::OpenAiResponses, credentials)
    }

    /// OpenAI Chat Completions and compatible servers such as LiteLLM and Ollama.
    ///
    /// Against hosts other than OpenAI's own the portable wire spellings are
    /// used (`max_tokens`, `stream_options` with only `include_usage`).
    /// OpenAI's `gpt-5.x` models reject function tools on this API unless a
    /// reasoning effort is set explicitly (`ReasoningConfig::Disabled` sends
    /// `reasoning_effort: none`). The Responses API has no such restriction.
    pub fn openai_chat(credentials: Credentials) -> Self {
        Self::new(ApiProfile::OpenAiChatCompletions, credentials)
    }

    /// OpenRouter's Chat Completions dialect.
    pub fn openrouter(credentials: Credentials) -> Self {
        Self::new(ApiProfile::OpenRouter, credentials)
    }

    /// ChatGPT subscription backend using the OpenAI Responses dialect.
    pub fn chatgpt(credentials: Credentials) -> Self {
        Self::new(ApiProfile::ChatGptResponses, credentials)
    }

    /// SpaceXAI's Responses dialect.
    pub fn xai(credentials: Credentials) -> Self {
        Self::new(ApiProfile::XaiResponses, credentials)
    }

    /// SpaceXAI's Chat Completions dialect.
    pub fn xai_chat(credentials: Credentials) -> Self {
        Self::new(ApiProfile::XaiChatCompletions, credentials)
    }

    /// Anthropic Messages API.
    pub fn anthropic(credentials: Credentials) -> Self {
        Self::new(ApiProfile::AnthropicMessages, credentials)
    }

    /// Gemini GenerateContent API.
    pub fn gemini(credentials: Credentials) -> Self {
        Self::new(ApiProfile::GeminiGenerateContent, credentials)
    }

    /// Anthropic models on Amazon Bedrock in `region`.
    ///
    /// Model ids are Bedrock's (`anthropic.claude-...` or an inference profile
    /// or ARN). Pass a Bedrock API key as [`Credentials::bearer`], or
    /// [`Credentials::none`] plus a SigV4 [`RequestAuthenticator`] through
    /// [`ProviderConfig::with_authenticator`].
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Configuration`](crate::ErrorKind::Configuration)
    /// when `region` is not an AWS region name.
    pub fn bedrock_anthropic(region: &str, credentials: Credentials) -> Result<Self> {
        let valid = !region.is_empty()
            && region
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-');
        if !valid {
            return Err(Error::configuration(format!(
                "`{region}` is not an AWS region name"
            )));
        }
        let base_url = Url::parse(&format!("https://bedrock-runtime.{region}.amazonaws.com"))
            .expect("a region name forms a valid host");
        Ok(Self::new(ApiProfile::BedrockAnthropic, credentials).with_base_url(base_url))
    }

    /// Use a custom API base URL.
    ///
    /// Any path prefix is preserved. The selected profile appends its request
    /// path exactly once.
    pub fn with_base_url(mut self, base_url: Url) -> Self {
        self.base_url = Some(base_url);
        self
    }

    /// Add a header to every request for this provider, replacing an earlier
    /// value with the same name.
    ///
    /// Per-request headers and authentication are applied later and can
    /// replace this value.
    pub fn with_header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.default_headers.insert(name, value);
        self
    }

    /// Replace the static credentials with a refreshable request authenticator.
    pub fn with_authenticator(mut self, authenticator: Arc<dyn RequestAuthenticator>) -> Self {
        self.authentication = Authentication::Authenticator(authenticator);
        self
    }

    /// The API behavior selected for this connection.
    pub fn profile(&self) -> ApiProfile {
        self.profile
    }

    pub(crate) fn custom_base_url(&self) -> Option<&Url> {
        self.base_url.as_ref()
    }

    pub(crate) fn default_headers(&self) -> &HeaderMap {
        &self.default_headers
    }

    pub(crate) fn authentication(&self) -> &Authentication {
        &self.authentication
    }
}

pub(crate) struct ProviderInner {
    pub(crate) config: ProviderConfig,
    pub(crate) transport: Arc<dyn HttpTransport>,
}

/// A configured provider connection. Cheap to clone.
#[derive(Clone)]
pub struct Provider {
    pub(crate) inner: Arc<ProviderInner>,
}

impl fmt::Debug for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Provider")
            .field("config", &self.inner.config)
            .finish()
    }
}

impl Provider {
    /// The API behavior used to encode and decode requests for this connection.
    pub fn profile(&self) -> ApiProfile {
        self.inner.config.profile()
    }

    /// The settings models on this provider accept. Profile defaults today.
    /// Per-model data will refine them here.
    pub fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::for_profile(self.profile())
    }

    /// The effective base URL (custom or the profile default).
    ///
    /// # Panics
    ///
    /// Panics if an internally defined profile default is not a valid URL.
    pub fn base_url(&self) -> Url {
        match self.inner.config.custom_base_url() {
            Some(url) => url.clone(),
            None => {
                Url::parse(self.profile().default_base_url()).expect("default base URLs are valid")
            }
        }
    }

    /// Create an execution handle for a model identifier.
    pub fn language_model(&self, id: impl Into<String>) -> LanguageModel {
        LanguageModel {
            provider: self.clone(),
            id: id.into(),
            capabilities: self.capabilities(),
        }
    }
}
