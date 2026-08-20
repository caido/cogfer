//! Generate results, finish reasons, warnings and response metadata.

use crate::message::{AssistantPart, Message, ReasoningPart, ToolCall};
use crate::metadata::ProviderMetadata;
use crate::usage::Usage;

/// Normalized reason a completion stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FinishReason {
    /// Natural end of turn.
    Stop,
    /// Output token limit or context window reached.
    Length,
    /// The model requested tool calls.
    ToolCalls,
    /// Provider content filter / refusal terminated output.
    ContentFilter,
    /// The provider paused the turn. Resume with the turn appended.
    Paused,
    /// The stream failed before finishing normally.
    Error,
    /// An unrecognized finish reason preserved in [`Finish::raw`].
    Other,
}

impl FinishReason {
    /// Whether this finish discards tool calls.
    pub(crate) fn discards_tool_calls(self) -> bool {
        matches!(
            self,
            FinishReason::Length | FinishReason::ContentFilter | FinishReason::Error
        )
    }
}

/// Finish reason plus the provider's raw value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finish {
    pub reason: FinishReason,
    /// The provider's untranslated finish/stop reason string.
    pub raw: Option<String>,
}

impl Finish {
    pub fn new(reason: FinishReason) -> Self {
        Self { reason, raw: None }
    }

    pub fn with_raw(reason: FinishReason, raw: impl Into<String>) -> Self {
        Self {
            reason,
            raw: Some(raw.into()),
        }
    }
}

/// Identifying metadata for one provider response.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResponseMetadata {
    /// Provider response id (`resp_...`, `msg_...`, `chatcmpl-...`).
    pub id: Option<String>,
    /// The concrete model that served the request, as reported by the provider.
    pub model: Option<String>,
    /// Provider request id from response headers, for support tickets.
    pub request_id: Option<String>,
}

/// Category of a [`Warning`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WarningKind {
    /// A request setting the selected API profile cannot express was dropped.
    UnsupportedSetting,
    /// Anything else worth surfacing without failing the request.
    Other,
}

/// A structured, non-fatal compatibility warning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Warning {
    pub kind: WarningKind,
    /// The setting the warning concerns (e.g. `"temperature"`).
    pub subject: Option<String>,
    pub message: String,
}

impl Warning {
    pub fn unsupported_setting(subject: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: WarningKind::UnsupportedSetting,
            subject: Some(subject.into()),
            message: message.into(),
        }
    }

    pub fn other(message: impl Into<String>) -> Self {
        Self {
            kind: WarningKind::Other,
            subject: None,
            message: message.into(),
        }
    }
}

/// The result of a non-streaming generate call (or of collecting a stream).
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateResult {
    /// Ordered assistant output parts.
    pub content: Vec<AssistantPart>,
    pub finish: Finish,
    pub usage: Usage,
    /// Warnings raised while lowering the request.
    pub warnings: Vec<Warning>,
    pub response: ResponseMetadata,
    /// Namespaced provider extras for the whole response.
    pub provider_metadata: ProviderMetadata,
}

impl GenerateResult {
    /// All text content concatenated in order.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for part in &self.content {
            if let AssistantPart::Text { text, .. } = part {
                out.push_str(text);
            }
        }
        out
    }

    /// All reasoning parts in order.
    pub fn reasoning(&self) -> impl Iterator<Item = &ReasoningPart> {
        self.content.iter().filter_map(|part| match part {
            AssistantPart::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
    }

    /// All tool calls in order.
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.content.iter().filter_map(|part| match part {
            AssistantPart::ToolCall(call) => Some(call),
            _ => None,
        })
    }

    /// All native compaction parts in provider order.
    pub fn compactions(&self) -> impl Iterator<Item = &crate::message::CompactionPart> {
        self.content.iter().filter_map(|part| match part {
            AssistantPart::Compaction(part) => Some(part),
            _ => None,
        })
    }

    /// Whether the model requested at least one tool call.
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls().next().is_some()
    }

    /// Parse the concatenated text as structured JSON output.
    ///
    /// # Errors
    ///
    /// Returns an error when the text content does not contain valid JSON for `T`.
    pub fn structured_output<T: serde::de::DeserializeOwned>(
        &self,
    ) -> Result<T, serde_json::Error> {
        let text = self.text();
        serde_json::from_str(&text)
    }

    /// Build the assistant history message for the next turn.
    pub fn to_assistant_message(&self) -> Message {
        Message::Assistant {
            content: self.content.clone(),
            provider_metadata: self.provider_metadata.clone(),
        }
    }
}
