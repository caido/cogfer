use serde::Deserialize;
use serde_json::{Value, json};

use super::SIGNATURE_KEY;
use crate::error::{Error, ErrorKind, Result};
use crate::http::{enrich_error_from_headers, error_kind_for_status};
use crate::message::{AssistantPart, ReasoningContent, ReasoningPart, ToolCall};
use crate::metadata::ProviderMetadata;
use crate::protocols::{ApiProfile, content_policy_kind, finalize_tool_calls};
use crate::response::{Finish, FinishReason, GenerateResult, ResponseMetadata};
use crate::transport::HeaderMap;
use crate::transport::HttpResponse;
use crate::usage::Usage;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiPart {
    #[serde(default)]
    pub(crate) text: Option<String>,
    #[serde(default)]
    pub(crate) thought: Option<bool>,
    #[serde(default)]
    pub(crate) thought_signature: Option<String>,
    #[serde(default)]
    pub(crate) function_call: Option<GeminiFunctionCall>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiFunctionCall {
    #[serde(default)]
    pub(crate) id: Option<String>,
    #[serde(default)]
    pub(crate) name: Option<String>,
    #[serde(default)]
    pub(crate) args: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiContent {
    #[serde(default)]
    pub(crate) parts: Option<Vec<GeminiPart>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiCandidate {
    #[serde(default)]
    pub(crate) content: Option<GeminiContent>,
    #[serde(default)]
    pub(crate) finish_reason: Option<String>,
    /// Search grounding preserved in `gemini.groundingMetadata`.
    #[serde(default)]
    pub(crate) grounding_metadata: Option<Value>,
    /// Legacy citation ranges preserved alongside grounding.
    #[serde(default)]
    pub(crate) citation_metadata: Option<Value>,
    #[serde(default)]
    pub(crate) safety_ratings: Option<Value>,
    #[serde(default)]
    pub(crate) finish_message: Option<String>,
}

impl GeminiCandidate {
    pub(crate) fn safety_metadata(&self) -> Option<ProviderMetadata> {
        if self.safety_ratings.is_none() && self.finish_message.is_none() {
            return None;
        }
        let mut metadata = serde_json::Map::new();
        if let Some(safety_ratings) = &self.safety_ratings {
            metadata.insert("safetyRatings".into(), safety_ratings.clone());
        }
        if let Some(finish_message) = &self.finish_message {
            metadata.insert("finishMessage".into(), finish_message.clone().into());
        }
        Some(ProviderMetadata::with("gemini", Value::Object(metadata)))
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PromptFeedback {
    #[serde(default)]
    pub(crate) block_reason: Option<String>,
    #[serde(default)]
    pub(crate) safety_ratings: Option<Value>,
}

impl PromptFeedback {
    pub(crate) fn provider_metadata(&self) -> ProviderMetadata {
        let mut feedback = serde_json::Map::new();
        if let Some(block_reason) = &self.block_reason {
            feedback.insert("blockReason".into(), block_reason.clone().into());
        }
        if let Some(safety_ratings) = &self.safety_ratings {
            feedback.insert("safetyRatings".into(), safety_ratings.clone());
        }
        ProviderMetadata::with("gemini", json!({"promptFeedback": feedback}))
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UsageMetadata {
    #[serde(default)]
    pub(crate) prompt_token_count: Option<u64>,
    #[serde(default)]
    pub(crate) candidates_token_count: Option<u64>,
    #[serde(default)]
    pub(crate) total_token_count: Option<u64>,
    #[serde(default)]
    pub(crate) thoughts_token_count: Option<u64>,
    #[serde(default)]
    pub(crate) cached_content_token_count: Option<u64>,
    #[serde(default)]
    pub(crate) tool_use_prompt_token_count: Option<u64>,
}

impl UsageMetadata {
    pub(crate) fn to_usage(&self) -> Usage {
        // Gemini excludes thinking tokens from its candidate count.
        let output_tokens = match (self.candidates_token_count, self.thoughts_token_count) {
            (Some(candidates), Some(thoughts)) => Some(candidates.saturating_add(thoughts)),
            (Some(candidates), None) => Some(candidates),
            (None, Some(thoughts)) => Some(thoughts),
            (None, None) => None,
        };
        // Gemini includes cached content in its prompt count.
        let mut usage = crate::protocols::normalize_usage(
            self.prompt_token_count,
            output_tokens,
            self.total_token_count,
            self.cached_content_token_count,
            None,
            self.thoughts_token_count,
        );
        usage.tool_use_prompt_tokens = self.tool_use_prompt_token_count;
        usage
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GenerateContentResponse {
    /// A Google RPC status: streams report failures as one such frame.
    #[serde(default)]
    pub(crate) error: Option<GoogleStatus>,
    #[serde(default)]
    pub(crate) candidates: Option<Vec<GeminiCandidate>>,
    #[serde(default)]
    pub(crate) prompt_feedback: Option<PromptFeedback>,
    #[serde(default)]
    pub(crate) usage_metadata: Option<UsageMetadata>,
    #[serde(default)]
    pub(crate) model_version: Option<String>,
    #[serde(default)]
    pub(crate) response_id: Option<String>,
}

pub(crate) fn signature_metadata(signature: &str) -> ProviderMetadata {
    ProviderMetadata::with("gemini", json!({SIGNATURE_KEY: signature}))
}

pub(crate) fn flush_thoughts(
    content: &mut Vec<AssistantPart>,
    buffer: &mut Vec<ReasoningContent>,
    metadata: &mut ProviderMetadata,
) {
    if buffer.is_empty() && metadata.is_empty() {
        return;
    }
    content.push(AssistantPart::Reasoning(ReasoningPart {
        id: None,
        content: std::mem::take(buffer),
        provider_metadata: std::mem::take(metadata),
    }));
}

/// Build the tool call for a `functionCall` part. Gemini may omit call ids,
/// so a stable one is synthesized. The provider id, when present, doubles as
/// item and call id for replay.
pub(crate) fn decode_function_call(
    part: &GeminiPart,
    function_call: &GeminiFunctionCall,
) -> ToolCall {
    let provider_id = function_call.id.clone().filter(|id| !id.is_empty());
    let call_id = provider_id
        .clone()
        .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple()));
    ToolCall {
        call_id,
        item_id: provider_id.clone(),
        provider_call_id: provider_id,
        name: function_call.name.clone().unwrap_or_default(),
        arguments: function_call
            .args
            .as_ref()
            .map_or_else(|| "{}".to_string(), Value::to_string),
        provider_metadata: part
            .thought_signature
            .as_deref()
            .map(signature_metadata)
            .unwrap_or_default(),
    }
}

/// The error for a candidate that finished with an error reason.
pub(crate) fn error_finish_error(finish: &Finish) -> Error {
    Error::new(
        ErrorKind::Provider,
        format!(
            "gemini ended the response with {}",
            finish.raw.as_deref().unwrap_or("an error finish reason"),
        ),
    )
    .with_origin(ApiProfile::GeminiGenerateContent.as_str())
}

pub(crate) fn decode_part(
    part: &GeminiPart,
    content: &mut Vec<AssistantPart>,
    thought_buffer: &mut Vec<ReasoningContent>,
    thought_metadata: &mut ProviderMetadata,
) {
    if part.thought == Some(true) {
        if let Some(text) = &part.text {
            thought_buffer.push(ReasoningContent::Summary { text: text.clone() });
        }
        if let Some(signature) = &part.thought_signature {
            thought_metadata.merge(signature_metadata(signature));
        }
        return;
    }
    flush_thoughts(content, thought_buffer, thought_metadata);

    if let Some(function_call) = &part.function_call {
        content.push(AssistantPart::ToolCall(decode_function_call(
            part,
            function_call,
        )));
        return;
    }

    if let Some(text) = &part.text {
        if text.is_empty() && part.thought_signature.is_none() {
            return;
        }
        let provider_metadata = part
            .thought_signature
            .as_deref()
            .map(signature_metadata)
            .unwrap_or_default();
        content.push(AssistantPart::Text {
            text: text.clone(),
            provider_metadata,
        });
    } else if let Some(signature) = &part.thought_signature {
        // Gemini requires standalone thought signatures on replay.
        content.push(AssistantPart::Text {
            text: String::new(),
            provider_metadata: signature_metadata(signature),
        });
    }
}

pub(crate) fn finish_from_reason(reason: Option<&str>) -> Finish {
    let Some(reason) = reason else {
        return Finish::new(FinishReason::Stop);
    };
    let mapped = match reason {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY"
        | "RECITATION"
        | "BLOCKLIST"
        | "PROHIBITED_CONTENT"
        | "SPII"
        | "IMAGE_SAFETY"
        | "IMAGE_PROHIBITED_CONTENT"
        | "ESCALATION" => FinishReason::ContentFilter,
        "MALFORMED_FUNCTION_CALL" | "MISSING_THOUGHT_SIGNATURE" | "MALFORMED_RESPONSE" => {
            FinishReason::Error
        }
        _ => FinishReason::Other,
    };
    Finish::with_raw(mapped, reason)
}

/// The `error` object of a Google RPC status envelope.
#[derive(Debug, Deserialize)]
pub(crate) struct GoogleStatus {
    /// The HTTP status the RPC status maps to.
    #[serde(default)]
    code: Option<u16>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    details: Vec<Value>,
}

impl GoogleStatus {
    fn detail_field(&self, detail_type: &str, field: &str) -> Option<&str> {
        self.details.iter().find_map(|detail| {
            (detail.get("@type").and_then(Value::as_str) == Some(detail_type))
                .then(|| detail.get(field).and_then(Value::as_str))
                .flatten()
        })
    }

    /// Classify and describe the failure. `http_status` is the transport
    /// status when the envelope arrived as an HTTP error body. Mid-stream
    /// frames rely on the envelope's own `code` instead.
    pub(crate) fn into_error(self, http_status: Option<u16>) -> Error {
        let status = http_status.or(self.code);
        let reason = self
            .detail_field("type.googleapis.com/google.rpc.ErrorInfo", "reason")
            .map(str::to_string);
        let retry_delay = self
            .detail_field("type.googleapis.com/google.rpc.RetryInfo", "retryDelay")
            .and_then(|delay| delay.strip_suffix('s'))
            .and_then(|seconds| seconds.parse::<f64>().ok())
            .and_then(|seconds| std::time::Duration::try_from_secs_f64(seconds).ok());
        let fallback = status.map_or(ErrorKind::Provider, error_kind_for_status);
        let kind = match self.status.as_deref() {
            Some("UNAUTHENTICATED") => ErrorKind::Authentication,
            Some("PERMISSION_DENIED") => {
                if reason.as_deref() == Some("API_KEY_INVALID") {
                    ErrorKind::Authentication
                } else {
                    ErrorKind::Permission
                }
            }
            Some("INVALID_ARGUMENT") => {
                if reason.as_deref() == Some("API_KEY_INVALID") {
                    ErrorKind::Authentication
                } else {
                    ErrorKind::InvalidRequest
                }
            }
            Some("FAILED_PRECONDITION") => ErrorKind::Permission,
            Some("NOT_FOUND") => ErrorKind::NotFound,
            Some("RESOURCE_EXHAUSTED") => ErrorKind::RateLimited,
            Some("UNAVAILABLE") => ErrorKind::Overloaded,
            Some("DEADLINE_EXCEEDED") => ErrorKind::Timeout,
            Some("INTERNAL") => ErrorKind::Provider,
            _ => fallback,
        };
        let kind = content_policy_kind(reason.as_deref().or(self.status.as_deref()), kind);

        let message = self
            .message
            .filter(|message| !message.is_empty())
            .unwrap_or_else(|| "gemini returned an error".to_string());
        let mut error =
            Error::new(kind, message).with_origin(ApiProfile::GeminiGenerateContent.as_str());
        if let Some(status) = status {
            error = error.with_status(status);
        }
        if let Some(code) = self.status.or(reason) {
            error = error.with_code(code);
        }
        if let Some(delay) = retry_delay {
            error = error.with_retry_after(delay);
        }
        error
    }
}

pub(crate) fn decode_gemini_error(status: u16, headers: &HeaderMap, body: &[u8]) -> Error {
    #[derive(Deserialize)]
    struct Envelope {
        #[serde(default)]
        error: Option<GoogleStatus>,
    }

    let error = match serde_json::from_slice::<Envelope>(body) {
        Ok(Envelope { error: Some(error) }) => error.into_error(Some(status)),
        _ => crate::protocols::fallback_provider_error(
            ApiProfile::GeminiGenerateContent,
            status,
            body,
        ),
    };
    enrich_error_from_headers(error, status, headers)
}

pub(crate) fn decode_gemini_response(response: &HttpResponse) -> Result<GenerateResult> {
    let parsed: GenerateContentResponse = serde_json::from_slice(&response.body)
        .map_err(|error| Error::malformed(format!("gemini: invalid response JSON: {error}")))?;

    if let Some(error) = parsed.error {
        return Err(error.into_error(None));
    }
    if parsed.candidates.as_ref().is_none_or(Vec::is_empty)
        && parsed.prompt_feedback.is_none()
        && parsed.usage_metadata.is_none()
        && parsed.response_id.is_none()
    {
        return Err(Error::malformed(
            "gemini: response body has no recognizable fields",
        ));
    }

    let mut content = Vec::new();
    let mut finish = Finish::new(FinishReason::Stop);
    let mut provider_metadata = ProviderMetadata::default();

    if let Some(candidate) = parsed
        .candidates
        .as_ref()
        .and_then(|candidates| candidates.first())
    {
        if let Some(grounding) = &candidate.grounding_metadata {
            provider_metadata.merge(ProviderMetadata::with(
                "gemini",
                json!({"groundingMetadata": grounding}),
            ));
        }
        if let Some(citations) = &candidate.citation_metadata {
            provider_metadata.merge(ProviderMetadata::with(
                "gemini",
                json!({"citationMetadata": citations}),
            ));
        }
        if let Some(metadata) = candidate.safety_metadata() {
            provider_metadata.merge(metadata);
        }
        let mut thought_buffer = Vec::new();
        let mut thought_metadata = ProviderMetadata::default();
        if let Some(parts) = candidate
            .content
            .as_ref()
            .and_then(|content| content.parts.as_ref())
        {
            for part in parts {
                decode_part(
                    part,
                    &mut content,
                    &mut thought_buffer,
                    &mut thought_metadata,
                );
            }
        }
        flush_thoughts(&mut content, &mut thought_buffer, &mut thought_metadata);
        finish = finish_from_reason(candidate.finish_reason.as_deref());
    } else if let Some(feedback) = &parsed.prompt_feedback {
        provider_metadata.merge(feedback.provider_metadata());
        finish = Finish::with_raw(
            FinishReason::ContentFilter,
            feedback
                .block_reason
                .clone()
                .unwrap_or_else(|| "BLOCKED".to_string()),
        );
    }

    if finish.reason == FinishReason::Error {
        return Err(error_finish_error(&finish));
    }
    finalize_tool_calls(&mut content, &mut finish, ApiProfile::GeminiGenerateContent)?;

    Ok(GenerateResult {
        content,
        finish,
        usage: parsed
            .usage_metadata
            .as_ref()
            .map(UsageMetadata::to_usage)
            .unwrap_or_default(),
        warnings: Vec::new(),
        response: ResponseMetadata {
            id: parsed.response_id,
            model: parsed.model_version,
            request_id: None,
        },
        provider_metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::{HeaderName, HeaderValue, header};

    #[test]
    fn error_headers_supply_request_id_and_retry_after() {
        let headers = HeaderMap::from_iter([
            (
                HeaderName::from_static("x-request-id"),
                HeaderValue::from_static("request-123"),
            ),
            (header::RETRY_AFTER, HeaderValue::from_static("7")),
        ]);
        let error = decode_gemini_error(429, &headers, b"not json");

        assert_eq!(error.request_id(), Some("request-123"));
        assert_eq!(error.retry_after(), Some(std::time::Duration::from_secs(7)));
    }

    #[test]
    fn stream_frame_errors_use_the_envelope_status() {
        let body = br#"{"error":{"code":503,"status":"UNAVAILABLE","message":"try later"}}"#;
        let parsed: GenerateContentResponse = serde_json::from_slice(body).unwrap();
        let error = parsed.error.unwrap().into_error(None);

        assert_eq!(error.kind(), ErrorKind::Overloaded);
        assert_eq!(error.status(), Some(503));
        assert!(error.retryable());
    }

    #[test]
    fn body_retry_info_takes_precedence_over_header() {
        let headers = HeaderMap::from_iter([(header::RETRY_AFTER, HeaderValue::from_static("7"))]);
        let body = br#"{"error":{"status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"2s"}]}}"#;
        let error = decode_gemini_error(429, &headers, body);

        assert_eq!(error.retry_after(), Some(std::time::Duration::from_secs(2)));
    }

    #[test]
    fn usage_output_tokens_saturate_when_provider_counts_overflow() {
        let metadata = UsageMetadata {
            candidates_token_count: Some(u64::MAX),
            thoughts_token_count: Some(1),
            ..UsageMetadata::default()
        };

        assert_eq!(metadata.to_usage().output_tokens, Some(u64::MAX));
    }
}
