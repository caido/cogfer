//! API profiles and their internal wire-protocol implementations.

mod error;
mod handler;
mod normalize;

use serde_json::json;

use crate::message::AssistantPart;
use crate::metadata::ProviderMetadata;
use crate::response::{Finish, FinishReason};

pub(crate) mod runner;
pub(crate) mod validate;

pub(crate) mod anthropic;
pub(crate) mod gemini;
pub(crate) mod openai;

pub(crate) use self::error::{
    content_policy_kind, fallback_provider_error, fallback_provider_error_message,
};
pub(crate) use self::handler::{
    LoweredRequest, ProtocolContext, ProtocolHandler, StreamDecoder, handler,
};
pub(crate) use self::normalize::normalize_usage;

/// Request, response, and streaming behavior for a provider API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ApiProfile {
    /// OpenAI Responses API (`POST {base}/responses`).
    OpenAiResponses,
    /// OpenAI Chat Completions API and compatible servers.
    OpenAiChatCompletions,
    /// OpenRouter's Chat Completions dialect with reasoning details, usage
    /// accounting and routing extensions.
    OpenRouter,
    /// The ChatGPT subscription backend using the Responses dialect.
    ChatGptResponses,
    /// xAI's Responses dialect (`POST {base}/responses`).
    XaiResponses,
    /// xAI's Chat Completions dialect.
    XaiChatCompletions,
    /// Anthropic Messages API (`POST {base}/messages`).
    AnthropicMessages,
    /// Gemini GenerateContent API.
    GeminiGenerateContent,
}

impl ApiProfile {
    /// Stable identifier (used in errors and warnings).
    pub fn as_str(self) -> &'static str {
        match self {
            ApiProfile::OpenAiResponses => "openai-responses",
            ApiProfile::OpenAiChatCompletions => "openai-chat",
            ApiProfile::OpenRouter => "openrouter",
            ApiProfile::ChatGptResponses => "chatgpt",
            ApiProfile::XaiResponses => "xai-responses",
            ApiProfile::XaiChatCompletions => "xai-chat",
            ApiProfile::AnthropicMessages => "anthropic",
            ApiProfile::GeminiGenerateContent => "gemini",
        }
    }

    /// The `provider_options` / `provider_metadata` namespace for this profile.
    pub fn namespace(self) -> &'static str {
        match self {
            ApiProfile::OpenAiResponses
            | ApiProfile::OpenAiChatCompletions
            | ApiProfile::ChatGptResponses
            | ApiProfile::XaiResponses
            | ApiProfile::XaiChatCompletions => "openai",
            ApiProfile::OpenRouter => "openrouter",
            ApiProfile::AnthropicMessages => "anthropic",
            ApiProfile::GeminiGenerateContent => "gemini",
        }
    }

    /// The official endpoint used when no custom base URL is configured.
    pub fn default_base_url(self) -> &'static str {
        match self {
            ApiProfile::OpenAiResponses | ApiProfile::OpenAiChatCompletions => {
                "https://api.openai.com/v1"
            }
            ApiProfile::OpenRouter => "https://openrouter.ai/api/v1",
            ApiProfile::ChatGptResponses => "https://chatgpt.com/backend-api/codex",
            ApiProfile::XaiResponses | ApiProfile::XaiChatCompletions => "https://api.x.ai/v1",
            ApiProfile::AnthropicMessages => "https://api.anthropic.com/v1",
            ApiProfile::GeminiGenerateContent => "https://generativelanguage.googleapis.com/v1beta",
        }
    }
}

impl std::fmt::Display for ApiProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Response-level provenance stamp: the profile that produced the turn.
///
/// Decoders record it in the library-reserved `caido-ai` namespace, and
/// request lowering uses [`foreign_origin`] to replay opaque state (encrypted
/// reasoning, item ids, server tool payloads) only to the profile that
/// produced it — backends cannot verify each other's items and reject the
/// request with errors like `invalid_encrypted_content`.
pub(crate) fn origin_metadata(profile: ApiProfile) -> ProviderMetadata {
    ProviderMetadata::with("caido-ai", json!({"profile": profile.as_str()}))
}

/// The producing profile stamped on a replayed turn, when it differs from the
/// serving profile. Unstamped turns count as native.
pub(crate) fn foreign_origin(metadata: &ProviderMetadata, profile: ApiProfile) -> Option<&str> {
    let stamp = metadata.get("caido-ai")?.get("profile")?.as_str()?;
    (stamp != profile.as_str()).then_some(stamp)
}

/// Apply the shared tool-call finish policy to a blocking result.
pub(crate) fn finalize_tool_calls(
    content: &mut Vec<AssistantPart>,
    finish: &mut Finish,
    profile: ApiProfile,
) -> crate::error::Result<()> {
    if finish.reason.discards_tool_calls() {
        content.retain(|part| !matches!(part, AssistantPart::ToolCall(_)));
        return Ok(());
    }

    let mut has_tool_calls = false;
    for part in content {
        let AssistantPart::ToolCall(call) = part else {
            continue;
        };
        call.normalize_blank_arguments();
        call.arguments_value().map_err(|source| {
            crate::error::Error::malformed(format!(
                "{profile}: tool call `{}` has invalid JSON arguments: {source}",
                call.call_id
            ))
            .with_source(source)
        })?;
        has_tool_calls = true;
    }
    if has_tool_calls && finish.reason == FinishReason::Stop {
        finish.reason = FinishReason::ToolCalls;
    }
    Ok(())
}
