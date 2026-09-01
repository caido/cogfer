use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::{Error, ErrorKind, Result};
use crate::http::{enrich_error_from_headers, error_kind_for_status};
use crate::message::{
    AssistantPart, CompactionPart, ProviderToolPart, ReasoningContent, ReasoningPart, ToolCall,
};
use crate::metadata::ProviderMetadata;
use crate::protocols::{ApiProfile, content_policy_kind, finalize_tool_calls};
use crate::response::{Finish, FinishReason, GenerateResult, ResponseMetadata};
use crate::transport::{HeaderMap, HttpResponse};
use crate::usage::Usage;

#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicUsageDetails {
    pub(crate) thinking_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicUsage {
    pub(crate) input_tokens: Option<u64>,
    pub(crate) output_tokens: Option<u64>,
    pub(crate) cache_creation_input_tokens: Option<u64>,
    pub(crate) cache_read_input_tokens: Option<u64>,
    pub(crate) output_tokens_details: Option<AnthropicUsageDetails>,
    /// Per-invocation usage for multi-pass turns. When present, the top-level
    /// counters describe only the final pass, so billing totals use this list.
    #[serde(default)]
    pub(crate) iterations: Vec<AnthropicUsage>,
}

impl AnthropicUsage {
    pub(crate) fn to_usage(&self) -> Usage {
        if !self.iterations.is_empty() {
            let mut total = Usage::default();
            for iteration in &self.iterations {
                total.add_from(&iteration.to_usage());
            }
            return total;
        }
        Usage {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            total_tokens: None,
            cached_input_tokens: self.cache_read_input_tokens,
            cache_creation_input_tokens: self.cache_creation_input_tokens,
            reasoning_tokens: self
                .output_tokens_details
                .as_ref()
                .and_then(|details| details.thinking_tokens),
            tool_use_prompt_tokens: None,
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ContentBlock {
    #[serde(rename = "type", default)]
    pub(crate) block_type: String,
    pub(crate) text: Option<String>,
    pub(crate) thinking: Option<String>,
    pub(crate) signature: Option<String>,
    pub(crate) data: Option<String>,
    pub(crate) id: Option<String>,
    pub(crate) name: Option<String>,
    pub(crate) input: Option<Value>,
    pub(crate) content: Option<Value>,
    /// Text block citations preserved in part metadata.
    pub(crate) citations: Option<Vec<Value>>,
    /// Everything else (server-tool block fields like `tool_use_id`), so
    /// unmodeled blocks reconstruct losslessly.
    #[serde(flatten)]
    pub(crate) extra: serde_json::Map<String, Value>,
}

/// Identify provider-executed content blocks by their type suffix.
pub(crate) fn is_server_tool_block(block_type: &str) -> bool {
    block_type.ends_with("_tool_use") || block_type.ends_with("_tool_result")
}

impl ContentBlock {
    /// Rebuild the raw provider JSON of an unmodeled block from the typed
    /// fields plus the flattened remainder.
    pub(crate) fn to_raw(&self) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("type".into(), Value::String(self.block_type.clone()));
        if let Some(id) = &self.id {
            object.insert("id".into(), Value::String(id.clone()));
        }
        if let Some(name) = &self.name {
            object.insert("name".into(), Value::String(name.clone()));
        }
        if let Some(input) = &self.input {
            object.insert("input".into(), input.clone());
        }
        if let Some(content) = &self.content {
            object.insert("content".into(), content.clone());
        }
        for (key, value) in &self.extra {
            object.insert(key.clone(), value.clone());
        }
        Value::Object(object)
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct MessageObject {
    pub(crate) id: Option<String>,
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) content: Vec<ContentBlock>,
    pub(crate) stop_reason: Option<String>,
    pub(crate) usage: Option<AnthropicUsage>,
    pub(crate) context_management: Option<Value>,
    pub(crate) stop_details: Option<Value>,
}

pub(crate) fn decode_content_block(block: &ContentBlock) -> Option<AssistantPart> {
    match block.block_type.as_str() {
        "text" => Some(AssistantPart::Text {
            text: block.text.clone().unwrap_or_default(),
            provider_metadata: match block.citations.as_ref().filter(|list| !list.is_empty()) {
                Some(citations) => {
                    ProviderMetadata::with("anthropic", serde_json::json!({"citations": citations}))
                }
                None => ProviderMetadata::default(),
            },
        }),
        "thinking" => Some(AssistantPart::Reasoning(ReasoningPart {
            id: None,
            content: vec![ReasoningContent::Text {
                text: block.thinking.clone().unwrap_or_default(),
                signature: block.signature.clone().filter(|s| !s.is_empty()),
            }],
            provider_metadata: ProviderMetadata::default(),
        })),
        "redacted_thinking" => Some(AssistantPart::Reasoning(ReasoningPart {
            id: None,
            content: vec![ReasoningContent::Redacted {
                data: block.data.clone().unwrap_or_default(),
            }],
            provider_metadata: ProviderMetadata::default(),
        })),
        "tool_use" => Some(AssistantPart::ToolCall(ToolCall {
            call_id: block.id.clone().unwrap_or_default(),
            item_id: None,
            name: block.name.clone().unwrap_or_default(),
            arguments: block
                .input
                .as_ref()
                .map(|input| input.to_string())
                .unwrap_or_default(),
            provider_metadata: ProviderMetadata::default(),
        })),
        "compaction" => Some(AssistantPart::Compaction(CompactionPart {
            id: None,
            content: block
                .content
                .as_ref()
                .and_then(Value::as_str)
                .map(str::to_string),
            encrypted_content: None,
        })),
        other => {
            // Anthropic correlates server tool results with these blocks on replay.
            is_server_tool_block(other).then(|| AssistantPart::ProviderTool {
                provider_tool: ProviderToolPart {
                    id: block.id.clone(),
                    kind: other.to_string(),
                    namespace: "anthropic".into(),
                    payload: block.to_raw(),
                },
            })
        }
    }
}

pub(crate) fn finish_from_stop_reason(stop_reason: Option<&str>) -> Finish {
    let Some(reason) = stop_reason else {
        return Finish::new(FinishReason::Stop);
    };
    let mapped = match reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" | "model_context_window_exceeded" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "refusal" => FinishReason::ContentFilter,
        "pause_turn" | "compaction" => FinishReason::Paused,
        _ => FinishReason::Other,
    };
    Finish::with_raw(mapped, reason)
}

pub(crate) fn decode_anthropic_response(response: &HttpResponse) -> Result<GenerateResult> {
    let parsed: MessageObject = serde_json::from_slice(&response.body)
        .map_err(|error| Error::malformed(format!("anthropic: invalid response JSON: {error}")))?;

    if parsed.id.is_none()
        && parsed.content.is_empty()
        && parsed.stop_reason.is_none()
        && parsed.usage.is_none()
    {
        return Err(Error::malformed(
            "anthropic: response body has no recognizable fields",
        ));
    }

    let mut content: Vec<_> = parsed
        .content
        .iter()
        .filter_map(decode_content_block)
        .collect();
    let mut finish = finish_from_stop_reason(parsed.stop_reason.as_deref());
    finalize_tool_calls(&mut content, &mut finish, ApiProfile::AnthropicMessages)?;

    let mut provider_metadata = ProviderMetadata::default();
    if let Some(context_management) = &parsed.context_management {
        provider_metadata.merge(ProviderMetadata::with(
            "anthropic",
            json!({"context_management": context_management}),
        ));
    }
    if let Some(stop_details) = &parsed.stop_details {
        provider_metadata.merge(ProviderMetadata::with(
            "anthropic",
            json!({"stop_details": stop_details}),
        ));
    }

    Ok(GenerateResult {
        content,
        finish,
        usage: parsed
            .usage
            .as_ref()
            .map(AnthropicUsage::to_usage)
            .unwrap_or_default(),
        warnings: Vec::new(),
        response: ResponseMetadata {
            id: parsed.id,
            model: parsed.model,
            request_id: None,
        },
        provider_metadata,
    })
}

/// Classify an Anthropic error object by its `type`, falling back to
/// `fallback` for unknown types. Shared by the HTTP and stream error paths so
/// both report the same kind for the same provider error.
pub(crate) fn anthropic_error_kind(
    error_type: Option<&str>,
    message: &str,
    fallback: ErrorKind,
) -> ErrorKind {
    let kind = match error_type {
        Some("authentication_error") => ErrorKind::Authentication,
        Some("permission_error" | "billing_error") => ErrorKind::Permission,
        Some("not_found_error") => ErrorKind::NotFound,
        Some("rate_limit_error") => ErrorKind::RateLimited,
        Some("overloaded_error") => ErrorKind::Overloaded,
        Some("timeout_error") => ErrorKind::Timeout,
        Some("api_error") => ErrorKind::Provider,
        Some("request_too_large") => ErrorKind::InvalidRequest,
        Some("invalid_request_error") => {
            if message.contains("prompt is too long")
                || message.contains("context window")
                || message.contains("context length")
            {
                ErrorKind::ContextLength
            } else {
                ErrorKind::InvalidRequest
            }
        }
        _ => fallback,
    };
    content_policy_kind(error_type, kind)
}

pub(crate) fn decode_anthropic_error(status: u16, headers: &HeaderMap, body: &[u8]) -> Error {
    #[derive(Deserialize)]
    struct Envelope {
        error: Option<ErrorBody>,
        request_id: Option<String>,
    }
    #[derive(Deserialize)]
    struct ErrorBody {
        #[serde(rename = "type", default)]
        error_type: Option<String>,
        message: Option<String>,
    }

    let parsed: Option<Envelope> = serde_json::from_slice(body).ok();
    let error_type = parsed
        .as_ref()
        .and_then(|envelope| envelope.error.as_ref())
        .and_then(|error| error.error_type.clone());
    let parsed_message = parsed
        .as_ref()
        .and_then(|envelope| envelope.error.as_ref())
        .and_then(|error| error.message.clone());
    let kind = anthropic_error_kind(
        error_type.as_deref(),
        parsed_message.as_deref().unwrap_or_default(),
        error_kind_for_status(status),
    );

    let message = parsed_message.unwrap_or_else(|| {
        crate::protocols::fallback_provider_error_message(
            ApiProfile::AnthropicMessages,
            status,
            body,
        )
    });
    let mut error = Error::new(kind, message).with_origin(ApiProfile::AnthropicMessages.as_str());
    if let Some(error_type) = error_type {
        error = error.with_code(error_type);
    }
    if let Some(request_id) = parsed.and_then(|envelope| envelope.request_id) {
        error = error.with_request_id(request_id);
    }
    enrich_error_from_headers(error, status, headers)
}
