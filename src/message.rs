//! The common message model shared by every protocol.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::metadata::ProviderMetadata;

/// A conversation message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Message {
    /// End-user input.
    User { content: Vec<UserPart> },
    /// A prior model turn, replayed for context.
    Assistant {
        content: Vec<AssistantPart>,
        /// Provider metadata for the whole turn. Decoders stamp the producing
        /// profile here (`{"caido-ai": {"profile": ...}}`) so lowering can
        /// drop opaque state another provider cannot verify. Preserve it when
        /// rebuilding history.
        #[serde(default, skip_serializing_if = "ProviderMetadata::is_empty")]
        provider_metadata: ProviderMetadata,
    },
    /// Results for tool calls issued by a prior assistant turn.
    Tool { content: Vec<ToolResultPart> },
}

impl Message {
    /// Convenience constructor for a plain-text user message.
    pub fn user(text: impl Into<String>) -> Self {
        Message::User {
            content: vec![UserPart::Text { text: text.into() }],
        }
    }

    /// Convenience constructor for a plain-text assistant message.
    pub fn assistant(text: impl Into<String>) -> Self {
        Message::Assistant {
            content: vec![AssistantPart::Text {
                text: text.into(),
                provider_metadata: ProviderMetadata::default(),
            }],
            provider_metadata: ProviderMetadata::default(),
        }
    }

    /// Convenience constructor for a single-result tool message.
    pub fn tool_result(part: ToolResultPart) -> Self {
        Message::Tool {
            content: vec![part],
        }
    }
}

/// Content inside a [`Message::User`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
#[non_exhaustive]
pub enum UserPart {
    Text { text: String },
}

/// Content inside a [`Message::Assistant`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
#[non_exhaustive]
pub enum AssistantPart {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "ProviderMetadata::is_empty")]
        provider_metadata: ProviderMetadata,
    },
    Reasoning(ReasoningPart),
    ToolCall(ToolCall),
    /// A provider-produced compaction item that replaces earlier history.
    Compaction(CompactionPart),
    /// A provider-executed tool item preserved for replay.
    ProviderTool {
        provider_tool: ProviderToolPart,
    },
}

/// A native-compaction item produced by the provider.
///
/// Anthropic uses [`CompactionPart::content`], while OpenAI Responses uses
/// [`CompactionPart::encrypted_content`]. Replay the part unmodified.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionPart {
    /// Provider item id (OpenAI `cmp_...`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Human-readable summary (Anthropic).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Opaque compacted state (OpenAI Responses).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
}

/// A provider-executed tool item preserved as exact JSON.
///
/// Matching API profiles replay `payload` verbatim. Others skip it with a warning.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderToolPart {
    /// Provider item/block id, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Provider item or block type.
    pub kind: String,
    /// ApiProfile namespace that produced and can replay the payload
    /// (`"openai"`, `"anthropic"`).
    pub namespace: String,
    /// The raw provider JSON, replayed verbatim.
    pub payload: Value,
}

/// A reasoning block emitted by a model, preserved for replay.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReasoningPart {
    /// Provider block or item id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Typed reasoning content in provider order.
    pub content: Vec<ReasoningContent>,
    /// Namespaced provider extras required for replay
    /// (e.g. OpenRouter `reasoning_details`).
    #[serde(default, skip_serializing_if = "ProviderMetadata::is_empty")]
    pub provider_metadata: ProviderMetadata,
}

impl ReasoningPart {
    /// The human-visible reasoning text (text + summary content, in order).
    pub fn visible_text(&self) -> String {
        let mut out = String::new();
        for content in &self.content {
            match content {
                ReasoningContent::Text { text, .. } | ReasoningContent::Summary { text } => {
                    out.push_str(text);
                }
                _ => {}
            }
        }
        out
    }
}

/// One piece of reasoning content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
#[non_exhaustive]
pub enum ReasoningContent {
    /// Raw reasoning text, optionally signed (Anthropic thinking signature,
    /// Gemini thought signature).
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Provider-generated reasoning summary (OpenAI Responses).
    Summary { text: String },
    /// Encrypted reasoning payload for stateless replay (OpenAI Responses
    /// `encrypted_content`).
    Encrypted { data: String },
    /// Safety-redacted reasoning (Anthropic `redacted_thinking`). Replay verbatim.
    Redacted { data: String },
}

/// A tool call requested by the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Stable consumer correlation id, synthesized when the provider omits one.
    /// Tool results reference this id.
    pub call_id: String,
    /// Provider item/block id (OpenAI Responses `fc_...`, Anthropic
    /// `toolu_...`, Chat Completions `call_...`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    /// Secondary provider correlation id where the protocol distinguishes it
    /// from the item id (OpenAI Responses `call_...`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_call_id: Option<String>,
    pub name: String,
    /// Raw provider JSON arguments. A blank
    /// value (zero-argument call) is normalized to `{}`.
    pub arguments: String,
    /// Namespaced provider extras (e.g. Gemini `thoughtSignature`).
    #[serde(default, skip_serializing_if = "ProviderMetadata::is_empty")]
    pub provider_metadata: ProviderMetadata,
}

impl ToolCall {
    /// Normalize blank zero-argument calls to `{}`.
    pub(crate) fn normalize_blank_arguments(&mut self) {
        if self.arguments.trim().is_empty() {
            self.arguments = "{}".to_string();
        }
    }

    /// Parse the raw argument JSON into a typed value.
    ///
    /// # Errors
    ///
    /// Returns an error when the arguments do not contain valid JSON for `T`.
    pub fn parse_arguments<T: serde::de::DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.arguments)
    }

    /// Parse the raw arguments as a [`serde_json::Value`].
    ///
    /// # Errors
    ///
    /// Returns an error when the provider produced invalid JSON.
    pub fn arguments_value(&self) -> Result<Value, serde_json::Error> {
        self.parse_arguments()
    }
}

/// The result of executing one tool call, sent back to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultPart {
    /// Correlates with [`ToolCall::call_id`].
    pub call_id: String,
    /// Provider item/block id of the originating call, when known. Filled
    /// automatically from history when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    /// Secondary provider call id of the originating call, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_call_id: Option<String>,
    /// Tool name required by Gemini and filled from history when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub content: ToolResultContent,
    /// Marks the result as an execution error where the protocol supports it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
}

impl ToolResultPart {
    /// Text-result constructor referencing a prior [`ToolCall`]. Copies all
    /// three identities and the name so replay needs no history scan.
    pub fn for_call(call: &ToolCall, content: impl Into<String>) -> Self {
        Self {
            call_id: call.call_id.clone(),
            item_id: call.item_id.clone(),
            provider_call_id: call.provider_call_id.clone(),
            name: Some(call.name.clone()),
            content: ToolResultContent::Text {
                text: content.into(),
            },
            is_error: false,
        }
    }

    /// JSON-result constructor referencing a prior [`ToolCall`].
    pub fn json_for_call(call: &ToolCall, content: Value) -> Self {
        Self {
            content: ToolResultContent::Json { value: content },
            ..Self::for_call(call, String::new())
        }
    }

    #[must_use = "tool result modifiers return an updated value"]
    pub fn with_error(mut self, is_error: bool) -> Self {
        self.is_error = is_error;
        self
    }
}

/// Payload of a tool result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
#[non_exhaustive]
pub enum ToolResultContent {
    Text { text: String },
    Json { value: Value },
}

impl ToolResultContent {
    /// Render the content as a string, serializing JSON values.
    pub fn to_text(&self) -> String {
        match self {
            ToolResultContent::Text { text } => text.clone(),
            ToolResultContent::Json { value } => value.to_string(),
        }
    }
}
