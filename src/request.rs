//! The common request model and its builder.

use std::num::NonZeroU32;

use serde_json::Value;

use crate::message::Message;
use crate::metadata::ProviderMetadata;
use crate::transport::{HeaderMap, HeaderName, HeaderValue};

/// A protocol-independent language-model request.
///
/// Unset optional fields are not sent. A setting that a profile can safely
/// omit produces a structured [`crate::Warning`], while content or operations
/// that cannot be represented return an error. Model-specific support remains the
/// provider's authority, and this crate does not infer or coerce settings from
/// model identifiers.
///
/// `Debug` redacts [`Request::extra_headers`] values because callers may put
/// credentials there.
#[derive(Clone, Default, PartialEq)]
pub struct Request {
    /// System or developer instructions kept outside message history.
    pub system: Option<String>,
    /// Conversation messages.
    pub messages: Vec<Message>,
    /// Tools the model may call.
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: Option<ToolChoice>,
    /// Whether the model may issue several tool calls in one turn.
    pub parallel_tool_calls: Option<bool>,
    /// Reasoning/thinking configuration. `None` uses the provider default.
    pub reasoning: Option<ReasoningConfig>,
    /// JSON-schema constrained output.
    pub structured_output: Option<StructuredOutput>,
    /// Opt-in native context compaction (OpenAI Responses / Anthropic).
    pub compaction: Option<Compaction>,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub stop_sequences: Vec<String>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub seed: Option<u64>,
    /// Provider-specific request parameters merged according to
    /// [`ProviderMetadata::merge`].
    pub provider_options: ProviderMetadata,
    /// Extra HTTP headers for this request. Values are redacted from `Debug`.
    pub extra_headers: HeaderMap,
    /// When streaming, also emit [`crate::StreamEvent::Raw`] events carrying
    /// unrecognized provider payloads.
    pub include_raw_events: bool,
}

impl Request {
    pub fn builder() -> RequestBuilder {
        RequestBuilder::default()
    }

    /// The system prompt, or `None` when unset or blank.
    pub fn system_prompt(&self) -> Option<&str> {
        self.system
            .as_deref()
            .filter(|text| !text.trim().is_empty())
    }
}

impl std::fmt::Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let redacted_headers: Vec<(&str, &str)> = self
            .extra_headers
            .keys()
            .map(|name| (name.as_str(), "<redacted>"))
            .collect();
        f.debug_struct("Request")
            .field("system", &self.system)
            .field("messages", &self.messages)
            .field("tools", &self.tools)
            .field("tool_choice", &self.tool_choice)
            .field("parallel_tool_calls", &self.parallel_tool_calls)
            .field("reasoning", &self.reasoning)
            .field("structured_output", &self.structured_output)
            .field("compaction", &self.compaction)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("top_k", &self.top_k)
            .field("stop_sequences", &self.stop_sequences)
            .field("presence_penalty", &self.presence_penalty)
            .field("frequency_penalty", &self.frequency_penalty)
            .field("seed", &self.seed)
            .field("provider_options", &self.provider_options)
            .field("extra_headers", &redacted_headers)
            .field("include_raw_events", &self.include_raw_events)
            .finish()
    }
}

/// Builder for [`Request`].
#[derive(Debug, Default, Clone)]
#[must_use = "request builders do nothing until build is called"]
pub struct RequestBuilder {
    request: Request,
}

impl RequestBuilder {
    pub fn message(mut self, message: Message) -> Self {
        self.request.messages.push(message);
        self
    }

    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.request.messages.extend(messages);
        self
    }

    /// Append system instructions, separating repeated calls with a blank line.
    pub fn system(mut self, text: impl Into<String>) -> Self {
        let text = text.into();
        match &mut self.request.system {
            Some(existing) => {
                existing.push_str("\n\n");
                existing.push_str(&text);
            }
            slot => *slot = Some(text),
        }
        self
    }

    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.request.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        self.request.tools.extend(tools);
        self
    }

    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.request.tool_choice = Some(choice);
        self
    }

    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.request.parallel_tool_calls = Some(enabled);
        self
    }

    pub fn reasoning(mut self, config: ReasoningConfig) -> Self {
        self.request.reasoning = Some(config);
        self
    }

    pub fn structured_output(mut self, output: StructuredOutput) -> Self {
        self.request.structured_output = Some(output);
        self
    }

    pub fn compaction(mut self, compaction: Compaction) -> Self {
        self.request.compaction = Some(compaction);
        self
    }

    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.request.max_output_tokens = Some(tokens);
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.request.temperature = Some(temperature);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.request.top_p = Some(top_p);
        self
    }

    pub fn top_k(mut self, top_k: u32) -> Self {
        self.request.top_k = Some(top_k);
        self
    }

    pub fn stop_sequence(mut self, stop: impl Into<String>) -> Self {
        self.request.stop_sequences.push(stop.into());
        self
    }

    pub fn presence_penalty(mut self, penalty: f64) -> Self {
        self.request.presence_penalty = Some(penalty);
        self
    }

    pub fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.request.frequency_penalty = Some(penalty);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.request.seed = Some(seed);
        self
    }

    /// Add provider-specific parameters under an API-profile namespace.
    ///
    /// Use [`crate::ApiProfile::namespace`] rather than deriving a namespace
    /// from the provider or model name.
    pub fn provider_option(mut self, namespace: impl Into<String>, value: Value) -> Self {
        self.request
            .provider_options
            .merge(ProviderMetadata::with(namespace, value));
        self
    }

    pub fn extra_header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.request.extra_headers.insert(name, value);
        self
    }

    pub fn include_raw_events(mut self, include: bool) -> Self {
        self.request.include_raw_events = include;
        self
    }

    pub fn build(self) -> Request {
        self.request
    }
}

/// A tool the model may call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    /// JSON Schema for the arguments object.
    pub parameters: Value,
    /// Request provider-side strict schema validation where supported.
    pub strict: Option<bool>,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            parameters,
            strict: None,
        }
    }
}

/// How the model chooses tools.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ToolChoice {
    /// The model decides (provider default).
    Auto,
    /// The model must not call tools.
    None,
    /// The model must call at least one tool.
    Required,
    /// The model must call this specific tool.
    Tool { name: String },
}

/// Exclusive reasoning configuration. Unset preserves the provider default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReasoningConfig {
    /// Disable reasoning when the selected model supports an off mode.
    Disabled,
    /// Use the provider's discrete effort control.
    Effort {
        effort: ReasoningEffort,
        /// Override whether visible reasoning appears in the response.
        output: Option<ReasoningOutput>,
    },
    /// Use an exact provider reasoning-token budget.
    Budget {
        tokens: NonZeroU32,
        /// Override whether visible reasoning appears in the response.
        output: Option<ReasoningOutput>,
    },
}

impl ReasoningConfig {
    /// Request a discrete effort while preserving output visibility defaults.
    pub fn effort(effort: ReasoningEffort) -> Self {
        Self::Effort {
            effort,
            output: None,
        }
    }

    /// Request a token budget while preserving output visibility defaults.
    pub fn budget(tokens: NonZeroU32) -> Self {
        Self::Budget {
            tokens,
            output: None,
        }
    }
}

/// Whether enabled reasoning is visible in the response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReasoningOutput {
    /// Include provider-generated reasoning or summaries.
    Include,
    /// Omit visible reasoning where the API profile supports it.
    Omit,
}

/// Discrete reasoning effort. [`ReasoningEffort::as_str`] is the wire value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl ReasoningEffort {
    /// The wire string for this effort level.
    pub fn as_str(self) -> &'static str {
        match self {
            ReasoningEffort::Minimal => "minimal",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
            ReasoningEffort::XHigh => "xhigh",
            ReasoningEffort::Max => "max",
        }
    }
}

/// JSON-schema constrained output configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredOutput {
    /// Schema name (required by OpenAI protocols).
    pub name: String,
    pub description: Option<String>,
    /// The JSON Schema the output must satisfy.
    pub schema: Value,
    /// Request provider-side strict validation where supported.
    pub strict: bool,
}

impl StructuredOutput {
    pub fn new(name: impl Into<String>, schema: Value) -> Self {
        Self {
            name: name.into(),
            description: None,
            schema,
            strict: true,
        }
    }
}

/// Opt-in native context compaction.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Compaction {
    /// Input-token threshold that triggers compaction, when the API profile
    /// supports a trigger. `None` uses the provider default.
    pub trigger_input_tokens: Option<u64>,
    /// Stop the turn right after compaction instead of continuing.
    pub pause_after_compaction: Option<bool>,
    /// Custom summarization instructions, replacing the provider default.
    pub instructions: Option<String>,
}

impl Compaction {
    /// Enable compaction with provider defaults.
    pub fn enabled() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_system_calls_append_in_order() {
        let request = Request::builder()
            .message(Message::user("hi"))
            .system("first")
            .system("second")
            .build();
        assert_eq!(request.system_prompt(), Some("first\n\nsecond"));
        assert_eq!(request.messages.len(), 1);
    }

    #[test]
    fn replacing_the_history_keeps_the_system_prompt() {
        let mut request = Request::builder().system("stay").build();
        request.messages = vec![Message::user("hi")];
        assert_eq!(request.system_prompt(), Some("stay"));
    }

    #[test]
    fn blank_system_prompt_reads_as_absent() {
        let request = Request::builder().system("  \n ").build();
        assert!(request.system.is_some());
        assert_eq!(request.system_prompt(), None);
    }
}
