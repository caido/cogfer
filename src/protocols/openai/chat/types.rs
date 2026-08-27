use serde::Deserialize;
use serde_json::{Value, json};

use super::{ChatDialect, OutputTokenAccounting};
use crate::error::{Error, ErrorKind, Result};
use crate::http::enrich_error_from_headers;
use crate::message::{AssistantPart, ReasoningContent, ReasoningPart, ToolCall};
use crate::metadata::ProviderMetadata;
use crate::protocols::{ApiProfile, content_policy_kind, finalize_tool_calls};
use crate::response::{Finish, FinishReason, GenerateResult, ResponseMetadata};
use crate::transport::{HeaderMap, HttpResponse};
use crate::usage::Usage;

#[derive(Debug, Deserialize)]
pub(crate) struct ChatUsageDetailsPrompt {
    #[serde(default)]
    pub(crate) cached_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) cache_write_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatUsageDetailsCompletion {
    #[serde(default)]
    pub(crate) reasoning_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatUsage {
    #[serde(default)]
    pub(crate) prompt_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) completion_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) total_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) prompt_tokens_details: Option<ChatUsageDetailsPrompt>,
    #[serde(default)]
    pub(crate) completion_tokens_details: Option<ChatUsageDetailsCompletion>,
    #[serde(default, flatten)]
    pub(crate) provider_fields: serde_json::Map<String, Value>,
}

impl ChatUsage {
    pub(crate) fn to_usage(&self, accounting: OutputTokenAccounting) -> Usage {
        let details = self.prompt_tokens_details.as_ref();
        let reasoning = self
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens);
        let output = match accounting {
            OutputTokenAccounting::IncludesReasoning => self.completion_tokens,
            OutputTokenAccounting::ExcludesReasoning => match (self.completion_tokens, reasoning) {
                (None, None) => None,
                (completion, reasoning) => Some(
                    completion
                        .unwrap_or(0)
                        .saturating_add(reasoning.unwrap_or(0)),
                ),
            },
        };
        crate::protocols::normalize_usage(
            self.prompt_tokens,
            output,
            self.total_tokens,
            details.and_then(|details| details.cached_tokens),
            details.and_then(|details| details.cache_write_tokens),
            reasoning,
        )
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatToolCallFunction {
    #[serde(default)]
    pub(crate) name: Option<String>,
    /// Raw argument JSON as a string or compatible gateway object.
    #[serde(default, deserialize_with = "deserialize_arguments")]
    pub(crate) arguments: Option<String>,
}

fn deserialize_arguments<'de, D>(deserializer: D) -> std::result::Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(match value {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) => Some(text),
        Some(other) => Some(other.to_string()),
    })
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatToolCall {
    #[serde(default)]
    pub(crate) id: Option<String>,
    #[serde(default)]
    pub(crate) function: Option<ChatToolCallFunction>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatMessage {
    #[serde(default)]
    pub(crate) content: Option<String>,
    #[serde(default)]
    pub(crate) refusal: Option<String>,
    #[serde(default)]
    pub(crate) tool_calls: Option<Vec<ChatToolCall>>,
    /// Url citations (also OpenRouter web-plugin citations).
    #[serde(default)]
    pub(crate) annotations: Option<Vec<Value>>,
    /// DeepSeek-style plaintext reasoning on compatible servers.
    #[serde(default)]
    pub(crate) reasoning_content: Option<String>,
    /// OpenRouter / Ollama-style plaintext reasoning.
    #[serde(default)]
    pub(crate) reasoning: Option<String>,
    /// OpenRouter typed reasoning blocks.
    #[serde(default)]
    pub(crate) reasoning_details: Option<Vec<Value>>,
    /// Deprecated single-call form, still emitted by some compatible servers.
    #[serde(default)]
    pub(crate) function_call: Option<ChatToolCallFunction>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatChoice {
    #[serde(default)]
    pub(crate) message: Option<ChatMessage>,
    #[serde(default)]
    pub(crate) finish_reason: Option<String>,
    #[serde(default)]
    pub(crate) native_finish_reason: Option<String>,
    #[serde(default)]
    pub(crate) error: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletion {
    #[serde(default)]
    pub(crate) id: Option<String>,
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) choices: Vec<ChatChoice>,
    #[serde(default)]
    pub(crate) usage: Option<ChatUsage>,
    #[serde(default)]
    pub(crate) error: Option<Value>,
    #[serde(default)]
    pub(crate) provider: Option<Value>,
}

pub(crate) fn result_provider_metadata(
    provider: Option<&Value>,
    usage: Option<&ChatUsage>,
    dialect: ChatDialect,
) -> ProviderMetadata {
    if dialect != ChatDialect::OpenRouter {
        return ProviderMetadata::default();
    }
    let mut metadata = serde_json::Map::new();
    if let Some(provider) = provider.filter(|value| !value.is_null()) {
        metadata.insert("provider".into(), provider.clone());
    }
    if let Some(usage) = usage {
        let mut fields = usage.provider_fields.clone();
        fields.retain(|_, value| !value.is_null());
        if !fields.is_empty() {
            metadata.insert("usage".into(), Value::Object(fields));
        }
    }
    if metadata.is_empty() {
        ProviderMetadata::default()
    } else {
        ProviderMetadata::with("openrouter", Value::Object(metadata))
    }
}

pub(crate) fn refusal_metadata() -> ProviderMetadata {
    ProviderMetadata::with("openai", json!({"content_type": "refusal"}))
}

pub(crate) fn map_chat_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" | "function_call" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        "error" => FinishReason::Error,
        _ => FinishReason::Other,
    }
}

/// Build a reasoning part from OpenRouter `reasoning_details` plus plaintext fallbacks.
pub(crate) fn reasoning_part_from_details(
    details: Option<Vec<Value>>,
    plaintext: Option<String>,
    dialect: ChatDialect,
) -> Option<ReasoningPart> {
    let mut content = Vec::new();
    let mut metadata = ProviderMetadata::default();
    if let Some(details) = details
        && !details.is_empty()
    {
        for detail in &details {
            let detail_type = detail.get("type").and_then(Value::as_str).unwrap_or("");
            match detail_type {
                "reasoning.text" => content.push(ReasoningContent::Text {
                    text: detail
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    signature: detail
                        .get("signature")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                }),
                "reasoning.summary" => content.push(ReasoningContent::Summary {
                    text: detail
                        .get("summary")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                }),
                "reasoning.encrypted" => content.push(ReasoningContent::Encrypted {
                    data: detail
                        .get("data")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                }),
                _ => {}
            }
        }
        metadata.insert(
            dialect.profile().namespace(),
            json!({"reasoning_details": details}),
        );
    }
    if content.is_empty()
        && let Some(text) = plaintext
        && !text.is_empty()
    {
        content.push(ReasoningContent::Text {
            text,
            signature: None,
        });
    }
    if content.is_empty() && metadata.is_empty() {
        return None;
    }
    Some(ReasoningPart {
        id: None,
        content,
        provider_metadata: metadata,
    })
}

pub(crate) fn decode_chat_response(
    response: &HttpResponse,
    dialect: ChatDialect,
) -> Result<GenerateResult> {
    let parsed: ChatCompletion = serde_json::from_slice(&response.body).map_err(|e| {
        Error::malformed(format!(
            "{}: invalid response JSON: {e}",
            dialect.profile().as_str()
        ))
    })?;
    let provider_metadata =
        result_provider_metadata(parsed.provider.as_ref(), parsed.usage.as_ref(), dialect);

    if let Some(error) = &parsed.error {
        return Err(decode_inline_error(error, dialect));
    }

    let choice = parsed
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| Error::malformed("chat completion has no choices"))?;
    if let Some(error) = &choice.error {
        return Err(decode_inline_error(error, dialect));
    }
    let saw_refusal = choice
        .message
        .as_ref()
        .and_then(|message| message.refusal.as_deref())
        .is_some_and(|refusal| !refusal.is_empty());
    let mut content = choice
        .message
        .map_or_else(Vec::new, |message| decode_chat_message(message, dialect));
    let mut finish = if saw_refusal {
        Finish::with_raw(FinishReason::ContentFilter, "refusal")
    } else {
        decode_chat_finish(choice.finish_reason, choice.native_finish_reason)
    };
    finalize_tool_calls(&mut content, &mut finish, dialect.profile())?;

    Ok(GenerateResult {
        content,
        finish,
        usage: parsed
            .usage
            .as_ref()
            .map(|usage| usage.to_usage(dialect.output_token_accounting()))
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

fn decode_chat_message(message: ChatMessage, dialect: ChatDialect) -> Vec<AssistantPart> {
    let mut content = Vec::new();
    let plaintext_reasoning = message.reasoning_content.or(message.reasoning);
    if let Some(part) =
        reasoning_part_from_details(message.reasoning_details, plaintext_reasoning, dialect)
    {
        content.push(AssistantPart::Reasoning(part));
    }
    if let Some(text) = message.content
        && !text.is_empty()
    {
        content.push(AssistantPart::Text {
            text,
            provider_metadata: match message.annotations.filter(|list| !list.is_empty()) {
                Some(annotations) => ProviderMetadata::with(
                    "openai",
                    serde_json::json!({"annotations": annotations}),
                ),
                None => ProviderMetadata::default(),
            },
        });
    }
    if let Some(refusal) = message.refusal
        && !refusal.is_empty()
    {
        content.push(AssistantPart::Text {
            text: refusal,
            provider_metadata: refusal_metadata(),
        });
    }

    if let Some(function) = message.function_call
        && message.tool_calls.is_none()
    {
        content.push(AssistantPart::ToolCall(ToolCall {
            call_id: "call_0".to_string(),
            item_id: None,
            name: function.name.unwrap_or_default(),
            arguments: function.arguments.unwrap_or_default(),
            provider_metadata: ProviderMetadata::default(),
        }));
    }
    for (index, call) in message
        .tool_calls
        .unwrap_or_default()
        .into_iter()
        .enumerate()
    {
        let function = call.function.unwrap_or(ChatToolCallFunction {
            name: None,
            arguments: None,
        });
        content.push(AssistantPart::ToolCall(ToolCall {
            call_id: call.id.unwrap_or_else(|| format!("call_{index}")),
            item_id: None,
            name: function.name.unwrap_or_default(),
            arguments: function.arguments.unwrap_or_default(),
            provider_metadata: ProviderMetadata::default(),
        }));
    }
    content
}

fn decode_chat_finish(
    finish_reason: Option<String>,
    native_finish_reason: Option<String>,
) -> Finish {
    match finish_reason {
        Some(reason) => Finish::with_raw(
            map_chat_finish_reason(&reason),
            native_finish_reason.unwrap_or_else(|| reason.clone()),
        ),
        None => Finish::new(FinishReason::Stop),
    }
}

/// Merge matching `reasoning_details` deltas into replayable signed blocks.
/// Two consecutive signed entries without indices always stay separate, so
/// provider-side signature validation cannot fail on replay.
pub(crate) fn merge_reasoning_details(details: Vec<Value>) -> Vec<Value> {
    let mut merged: Vec<Value> = Vec::new();
    for detail in details {
        let detail_type = detail
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_string);
        let index = detail.get("index").and_then(Value::as_u64);
        let joinable = merged.last_mut().filter(|last| {
            last.get("type").and_then(Value::as_str).map(str::to_string) == detail_type
                && last.get("index").and_then(Value::as_u64) == index
                // Signed entries without indices are separate blocks.
                && !(index.is_none()
                    && last.get("signature").is_some_and(|value| !value.is_null())
                    && detail.get("signature").is_some_and(|value| !value.is_null()))
        });
        match joinable {
            Some(last) => {
                for key in ["text", "summary", "data"] {
                    if let Some(addition) = detail.get(key).and_then(Value::as_str) {
                        match last.get_mut(key) {
                            Some(Value::String(existing)) => existing.push_str(addition),
                            _ => last[key] = json!(addition),
                        }
                    }
                }
                for key in ["signature", "id", "format"] {
                    if let Some(value) = detail.get(key)
                        && !value.is_null()
                    {
                        last[key] = value.clone();
                    }
                }
            }
            None => merged.push(detail),
        }
    }
    merged
}

/// Decode an OpenRouter error response, whose `error` object carries
/// upstream details the shared OpenAI decoder does not know about.
pub(crate) fn decode_openrouter_error(status: u16, headers: &HeaderMap, body: &[u8]) -> Error {
    #[derive(Deserialize)]
    struct Envelope {
        error: Option<Value>,
    }
    let error = match serde_json::from_slice::<Envelope>(body) {
        Ok(Envelope { error: Some(error) }) => decode_inline_error(&error, ChatDialect::OpenRouter),
        _ => crate::protocols::fallback_provider_error(ApiProfile::OpenRouter, status, body),
    };
    enrich_error_from_headers(error, status, headers)
}

/// Decode an error object found inside an HTTP 200 body or stream chunk.
pub(crate) fn decode_inline_error(error: &Value, dialect: ChatDialect) -> Error {
    let mut message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("provider returned an error")
        .to_string();
    // OpenRouter often stores useful upstream details behind a generic message.
    if let Some(metadata) = error.get("metadata") {
        if let Some(raw) = metadata.get("raw").and_then(Value::as_str) {
            message = format!("{message}: {}", crate::util::truncate_for_error(raw, 300));
        }
        if let Some(provider_name) = metadata.get("provider_name").and_then(Value::as_str) {
            message = format!("{message} (provider: {provider_name})");
        }
    }
    let code = error.get("code");
    // Gateways also use this field for opaque numeric codes.
    let status = code
        .and_then(Value::as_u64)
        .and_then(|code| u16::try_from(code).ok())
        .filter(|code| (100..=599).contains(code));
    let metadata_type = error
        .get("metadata")
        .and_then(|metadata| metadata.get("error_type"))
        .and_then(Value::as_str);

    let kind = match metadata_type {
        Some("context_length_exceeded" | "max_tokens_exceeded" | "token_limit_exceeded") => {
            ErrorKind::ContextLength
        }
        Some("authentication") => ErrorKind::Authentication,
        Some("permission_denied" | "payment_required") => ErrorKind::Permission,
        Some("rate_limit_exceeded") => ErrorKind::RateLimited,
        Some("provider_overloaded" | "provider_unavailable") => ErrorKind::Overloaded,
        Some("timeout") => ErrorKind::Timeout,
        _ => status
            .map(crate::http::error_kind_for_status)
            .unwrap_or(ErrorKind::Provider),
    };
    let string_code = code.and_then(Value::as_str);
    let kind = content_policy_kind(metadata_type.or(string_code), kind);

    let mut out = Error::new(kind, message).with_origin(dialect.profile().as_str());
    if let Some(status) = status {
        out = out.with_status(status);
    }
    if let Some(code) = code {
        match code {
            Value::String(code) => out = out.with_code(code.clone()),
            Value::Number(code) => out = out.with_code(code.to_string()),
            _ => {}
        }
    }
    if let Some(error_type) = metadata_type {
        out = out.with_code(error_type.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::{HeaderName, HeaderValue, header};

    #[test]
    fn non_json_openrouter_error_keeps_header_metadata() {
        let headers = HeaderMap::from_iter([
            (
                HeaderName::from_static("x-request-id"),
                HeaderValue::from_static("request-123"),
            ),
            (header::RETRY_AFTER, HeaderValue::from_static("4")),
        ]);
        let error = decode_openrouter_error(503, &headers, b"upstream unavailable");

        assert_eq!(error.request_id(), Some("request-123"));
        assert_eq!(error.retry_after(), Some(std::time::Duration::from_secs(4)));
    }
}
