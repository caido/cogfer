use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::{Error, ErrorKind, Result};
use crate::message::{
    AssistantPart, CompactionPart, ProviderToolPart, ReasoningContent, ReasoningPart, ToolCall,
};
use crate::metadata::ProviderMetadata;
use crate::protocols::{
    ApiProfile, ProtocolContext, content_policy_kind, finalize_tool_calls, origin_metadata,
};
use crate::response::{Finish, FinishReason, GenerateResult, ResponseMetadata};
use crate::transport::HttpResponse;
use crate::usage::Usage;

#[derive(Debug, Deserialize)]
pub(crate) struct ResponseUsageDetailsInput {
    pub(crate) cached_tokens: Option<u64>,
    pub(crate) cache_write_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ResponseUsageDetailsOutput {
    pub(crate) reasoning_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ResponseUsage {
    pub(crate) input_tokens: Option<u64>,
    pub(crate) output_tokens: Option<u64>,
    pub(crate) total_tokens: Option<u64>,
    pub(crate) input_tokens_details: Option<ResponseUsageDetailsInput>,
    pub(crate) output_tokens_details: Option<ResponseUsageDetailsOutput>,
}

impl ResponseUsage {
    pub(crate) fn to_usage(&self) -> Usage {
        let details = self.input_tokens_details.as_ref();
        crate::protocols::normalize_usage(
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
            details.and_then(|details| details.cached_tokens),
            details.and_then(|details| details.cache_write_tokens),
            self.output_tokens_details
                .as_ref()
                .and_then(|details| details.reasoning_tokens),
        )
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ResponseError {
    pub(crate) code: Option<String>,
    pub(crate) message: Option<String>,
}

/// The error a response object carries when generation failed.
pub(crate) fn failed_response_error(error: Option<ResponseError>, profile: ApiProfile) -> Error {
    let (code, message) = error
        .map(|error| (error.code, error.message))
        .unwrap_or_default();
    let kind = content_policy_kind(code.as_deref(), ErrorKind::Provider);
    let mut error = Error::new(
        kind,
        message.unwrap_or_else(|| "response generation failed".into()),
    )
    .with_origin(profile.as_str());
    if let Some(code) = code {
        error = error.with_code(code);
    }
    error
}

/// A top-level `error` stream event. The API keeps `code`/`message` at the
/// top level while the ChatGPT backend nests them under `error`.
#[derive(Debug, Deserialize)]
pub(crate) struct StreamErrorEvent {
    pub(crate) code: Option<Value>,
    pub(crate) message: Option<String>,
    pub(crate) error: Option<Value>,
}

impl StreamErrorEvent {
    pub(crate) fn into_error(self, profile: ApiProfile) -> Error {
        let nested = self.error.as_ref();
        let code = match self
            .code
            .or_else(|| nested.and_then(|error| error.get("code").cloned()))
        {
            Some(Value::String(code)) => Some(code),
            Some(Value::Number(code)) => Some(code.to_string()),
            _ => None,
        };
        let message = self
            .message
            .or_else(|| {
                nested
                    .and_then(|error| error.get("message"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "stream error event".into());
        let kind = content_policy_kind(code.as_deref(), ErrorKind::Provider);
        let mut error = Error::new(kind, message).with_origin(profile.as_str());
        if let Some(code) = code {
            error = error.with_code(code);
        }
        error
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct IncompleteDetails {
    pub(crate) reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ItemContentPart {
    #[serde(rename = "type", default)]
    pub(crate) part_type: String,
    pub(crate) text: Option<String>,
    pub(crate) refusal: Option<String>,
    /// Server-tool citations preserved in the text part's provider metadata.
    pub(crate) annotations: Option<Vec<Value>>,
}

/// Whether an item represents a provider-executed tool.
pub(crate) fn is_server_tool_item(item_type: &str) -> bool {
    (item_type.ends_with("_call") && item_type != "function_call") || item_type.starts_with("mcp_")
}

#[derive(Debug, Deserialize)]
pub(crate) struct SummaryPart {
    pub(crate) text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OutputItem {
    #[serde(rename = "type", default)]
    pub(crate) item_type: String,
    pub(crate) status: Option<String>,
    pub(crate) id: Option<String>,
    pub(crate) call_id: Option<String>,
    pub(crate) name: Option<String>,
    pub(crate) arguments: Option<String>,
    pub(crate) content: Option<Vec<ItemContentPart>>,
    pub(crate) summary: Option<Vec<SummaryPart>>,
    pub(crate) encrypted_content: Option<String>,
    /// Assistant-message phase (`commentary` / `final_answer`) preserved for replay.
    pub(crate) phase: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ResponseObject {
    pub(crate) id: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) status: Option<String>,
    pub(crate) error: Option<ResponseError>,
    pub(crate) incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    pub(crate) output: Vec<Value>,
    pub(crate) usage: Option<ResponseUsage>,
}

fn decode_message_item(item: OutputItem) -> Vec<AssistantPart> {
    let mut parts = Vec::new();
    let phase_metadata = item
        .phase
        .as_ref()
        .map(|phase| ProviderMetadata::with("openai", json!({"phase": phase})))
        .unwrap_or_default();
    for content in item.content.unwrap_or_default() {
        match content.part_type.as_str() {
            "output_text" => parts.push(AssistantPart::Text {
                text: content.text.unwrap_or_default(),
                provider_metadata: {
                    let mut metadata = phase_metadata.clone();
                    if let Some(annotations) = content.annotations.filter(|list| !list.is_empty()) {
                        metadata.merge(ProviderMetadata::with(
                            "openai",
                            json!({"annotations": annotations}),
                        ));
                    }
                    metadata
                },
            }),
            "refusal" => parts.push(AssistantPart::Text {
                text: content.refusal.unwrap_or_default(),
                provider_metadata: {
                    let mut metadata = phase_metadata.clone();
                    metadata.merge(ProviderMetadata::with(
                        "openai",
                        json!({"content_type": "refusal"}),
                    ));
                    metadata
                },
            }),
            _ => {}
        }
    }
    parts
}

pub(crate) fn is_refusal_part(part: &AssistantPart) -> bool {
    let AssistantPart::Text {
        provider_metadata, ..
    } = part
    else {
        return false;
    };
    provider_metadata
        .get("openai")
        .and_then(|namespace| namespace.get("content_type"))
        .and_then(Value::as_str)
        == Some("refusal")
}

/// Decode one output item, or `None` when it carries nothing to replay.
pub(crate) fn decode_output_item(raw: &Value) -> Result<Option<Vec<AssistantPart>>> {
    let item = OutputItem::deserialize(raw).map_err(|source| {
        Error::malformed("openai-responses: invalid output item").with_source(source)
    })?;
    match item.item_type.as_str() {
        "message" => Ok(Some(decode_message_item(item))),
        "function_call" => {
            // `status` is optional on the wire. Drop only explicitly
            // non-completed calls.
            if item
                .status
                .as_deref()
                .is_some_and(|status| status != "completed")
            {
                return Ok(None);
            }
            let call_id = item.call_id.ok_or_else(|| {
                Error::malformed("openai-responses: function_call item missing call_id")
            })?;
            Ok(Some(vec![AssistantPart::ToolCall(ToolCall {
                call_id,
                item_id: item.id.filter(|id| !id.is_empty()),
                name: item.name.unwrap_or_default(),
                arguments: item.arguments.unwrap_or_default(),
                provider_metadata: ProviderMetadata::default(),
            })]))
        }
        "reasoning" => {
            let mut content = Vec::new();
            for summary in item.summary.unwrap_or_default() {
                if let Some(text) = summary.text {
                    content.push(ReasoningContent::Summary { text });
                }
            }
            for part in item.content.unwrap_or_default() {
                if part.part_type == "reasoning_text"
                    && let Some(text) = part.text
                {
                    content.push(ReasoningContent::Text {
                        text,
                        signature: None,
                    });
                }
            }
            if let Some(encrypted) = item.encrypted_content.filter(|data| !data.is_empty()) {
                content.push(ReasoningContent::Encrypted { data: encrypted });
            }
            Ok(Some(vec![AssistantPart::Reasoning(ReasoningPart {
                id: item.id,
                content,
                provider_metadata: ProviderMetadata::default(),
            })]))
        }
        "compaction" => Ok(Some(vec![AssistantPart::Compaction(CompactionPart {
            id: item.id,
            content: None,
            encrypted_content: item.encrypted_content,
        })])),
        other => {
            // Providers correlate server tool results with these raw items on replay.
            Ok(is_server_tool_item(other).then(|| {
                vec![AssistantPart::ProviderTool {
                    provider_tool: ProviderToolPart {
                        id: item.id,
                        kind: other.to_string(),
                        namespace: "openai".into(),
                        payload: raw.clone(),
                    },
                }]
            }))
        }
    }
}

pub(crate) fn finish_from_status(
    status: Option<&str>,
    incomplete: Option<&IncompleteDetails>,
) -> Finish {
    match status {
        Some("incomplete") => {
            let reason = incomplete.and_then(|details| details.reason.as_deref());
            match reason {
                Some("max_output_tokens") | Some("max_tokens") => {
                    Finish::with_raw(FinishReason::Length, reason.unwrap_or_default())
                }
                Some("content_filter") => {
                    Finish::with_raw(FinishReason::ContentFilter, "content_filter")
                }
                Some(other) => Finish::with_raw(FinishReason::Other, other),
                None => Finish::with_raw(FinishReason::Other, "incomplete"),
            }
        }
        Some("completed") => Finish::with_raw(FinishReason::Stop, "completed"),
        Some(other) => Finish::with_raw(FinishReason::Other, other),
        None => Finish::new(FinishReason::Stop),
    }
}

pub(crate) fn decode_response_object(
    ctx: &ProtocolContext<'_>,
    parsed: ResponseObject,
) -> Result<GenerateResult> {
    if parsed.id.is_none()
        && parsed.status.is_none()
        && parsed.output.is_empty()
        && parsed.usage.is_none()
        && parsed.error.is_none()
    {
        return Err(
            Error::malformed("openai-responses: response body has no recognizable fields")
                .with_model(ctx.model.to_string()),
        );
    }
    if let Some(status @ ("queued" | "in_progress")) = parsed.status.as_deref() {
        return Err(Error::new(
            ErrorKind::Provider,
            format!(
                "{}: response is still `{status}`; background responses are not supported",
                ctx.profile.as_str()
            ),
        )
        .with_model(ctx.model.to_string()));
    }
    // A populated error object means the generation failed even when a proxy
    // answered with a 200 and no `failed` status.
    if parsed.status.as_deref() == Some("failed") || parsed.error.is_some() {
        return Err(failed_response_error(parsed.error, ctx.profile).with_model(ctx.model));
    }

    let mut content = Vec::new();
    for raw in &parsed.output {
        if let Some(parts) = decode_output_item(raw)? {
            content.extend(parts);
        }
    }
    let mut finish = if content.iter().any(is_refusal_part) {
        Finish::with_raw(FinishReason::ContentFilter, "refusal")
    } else {
        finish_from_status(parsed.status.as_deref(), parsed.incomplete_details.as_ref())
    };
    finalize_tool_calls(&mut content, &mut finish, ctx.profile)?;

    Ok(GenerateResult {
        finish,
        usage: parsed
            .usage
            .as_ref()
            .map(ResponseUsage::to_usage)
            .unwrap_or_default(),
        warnings: Vec::new(),
        response: ResponseMetadata {
            id: parsed.id,
            model: parsed.model,
            request_id: None,
        },
        provider_metadata: origin_metadata(ctx.profile),
        content,
    })
}

pub(crate) fn decode_openai_response(
    ctx: &ProtocolContext<'_>,
    response: &HttpResponse,
) -> Result<GenerateResult> {
    let parsed: ResponseObject = serde_json::from_slice(&response.body).map_err(|error| {
        Error::malformed(format!("openai-responses: invalid response JSON: {error}"))
    })?;
    decode_response_object(ctx, parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_output_item_is_an_error() {
        let error = decode_output_item(&json!({"type": "message", "content": "invalid"}))
            .expect_err("invalid item must not be silently dropped");

        assert_eq!(error.kind(), ErrorKind::MalformedResponse);
    }

    #[test]
    fn function_call_with_a_blank_id_has_no_item_id() {
        let parts = decode_output_item(&json!({
            "type": "function_call", "id": "", "call_id": "call_1",
            "name": "lookup", "arguments": "{}", "status": "completed",
        }))
        .unwrap()
        .unwrap();

        let AssistantPart::ToolCall(call) = &parts[0] else {
            panic!("expected a tool call");
        };
        assert_eq!(call.item_id, None);
    }

    #[test]
    fn function_call_without_call_id_is_an_error() {
        let error = decode_output_item(
            &json!({"type": "function_call", "status": "completed", "name": "lookup"}),
        )
        .expect_err("missing correlation id must not be silently dropped");

        assert_eq!(error.kind(), ErrorKind::MalformedResponse);
    }
}
