//! # llmwire
//!
//! A focused language-model SDK. Its API profiles normalize OpenAI Responses
//! and Chat Completions, ChatGPT subscriptions, SpaceXAI, OpenRouter, Anthropic
//! (directly and, with the `aws` feature, on Amazon Bedrock), and Gemini into
//! one request model, stream contract, and error taxonomy.
//!
//! ```no_run
//! # #[cfg(feature = "reqwest-transport")]
//! # async fn demo() -> Result<(), llmwire::Error> {
//! use llmwire::{Client, Credentials, ProviderConfig, Request, Message};
//!
//! // Once per process, unless the host already installed a Rustls provider.
//! llmwire::transport::install_default_crypto_provider();
//! let client = Client::builder().build()?;
//! let provider = client.provider(
//!     ProviderConfig::openai_responses(Credentials::api_key("sk-...")),
//! )?;
//! let model = provider.language_model("gpt-5");
//!
//! let request = Request::builder()
//!     .message(Message::user("Say hi"))
//!     .build();
//! let result = model.generate(request).await?;
//! println!("{}", result.text());
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub use self::auth::{Credentials, Rejection, RequestAuthenticator, SecretString, TokenStore};
pub use self::capabilities::{ModelCapabilities, ReasoningSupport};
pub use self::client::{Client, ClientBuilder};
pub use self::error::{Error, ErrorKind, Result};
pub use self::message::{
    AssistantPart, CompactionPart, Message, ProviderToolPart, ReasoningContent, ReasoningPart,
    ToolCall, ToolResultContent, ToolResultPart, UserPart,
};
pub use self::metadata::ProviderMetadata;
pub use self::model::LanguageModel;
pub use self::oauth::{DevicePoll, OAuthClientConfig, OAuthStatus};
pub use self::protocols::ApiProfile;
pub use self::provider::{Provider, ProviderConfig};
pub use self::request::{
    Compaction, ReasoningConfig, ReasoningEffort, ReasoningOutput, Request, RequestBuilder,
    StructuredOutput, ToolChoice, ToolDefinition,
};
pub use self::response::{
    Finish, FinishReason, GenerateResult, ResponseMetadata, Warning, WarningKind,
};
pub use self::stream::{Citation, EventStream, StreamAccumulator, StreamEvent};
pub use self::usage::Usage;

#[cfg(feature = "aws")]
pub mod aws;
pub mod oauth;
pub mod transport;

mod auth;
mod capabilities;
mod client;
mod error;
mod http;
mod message;
mod metadata;
mod model;
mod protocols;
mod provider;
mod request;
mod response;
mod stream;
mod usage;
mod util;
