use serde::Deserialize;
use serde_json::{Value, json};

use super::types::{
    ResponseObject, StreamErrorEvent, decode_output_item, failed_response_error,
    finish_from_status, is_server_tool_item,
};
use crate::error::{Error, Result};
use crate::message::{AssistantPart, ProviderToolPart};
use crate::metadata::ProviderMetadata;
use crate::protocols::{ApiProfile, StreamDecoder, origin_metadata};
use crate::response::{Finish, FinishReason, ResponseMetadata};
use crate::stream::{Citation, StreamEvent, StreamNormalizer};

pub(crate) struct ResponsesStreamDecoder {
    /// Profile used to attribute errors from this shared decoder.
    profile: ApiProfile,
    /// Open function call ids by item id.
    open_calls: std::collections::HashMap<String, String>,
    /// Open text block ids kept until item-level metadata arrives.
    open_texts: std::collections::HashMap<String, Vec<String>>,
    /// Pending annotations by text block id.
    annotations: std::collections::HashMap<String, Vec<Value>>,
    saw_refusal: bool,
    errored: bool,
    /// Whether the turn's origin-profile stamp was emitted.
    stamped: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamEnvelope {
    #[serde(rename = "type", default)]
    event_type: String,
    response: Option<Value>,
    item: Option<Value>,
    item_id: Option<String>,
    content_index: Option<u64>,
    delta: Option<String>,
    part: Option<Value>,
    annotation: Option<Value>,
    /// The `error` event's fields, unused by every other event type.
    #[serde(flatten)]
    error: StreamErrorEvent,
}

impl ResponsesStreamDecoder {
    pub(crate) fn new(profile: ApiProfile) -> Self {
        Self {
            profile,
            open_calls: Default::default(),
            open_texts: Default::default(),
            annotations: Default::default(),
            saw_refusal: false,
            errored: false,
            stamped: false,
        }
    }

    fn text_block_id(item_id: &str, content_index: Option<u64>) -> String {
        format!("{item_id}:{}", content_index.unwrap_or(0))
    }

    fn server_tool_status(event_type: &str) -> Option<&str> {
        let rest = event_type.strip_prefix("response.")?;
        let (head, status) = rest.rsplit_once('.')?;
        // Argument delta events are not server tool statuses.
        (is_server_tool_item(head) && !head.ends_with("_arguments")).then_some(status)
    }

    fn handle_response_metadata(
        response: Option<&Value>,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let Some(response) = response else {
            return;
        };
        let id = response.get("id").and_then(Value::as_str);
        let model = response.get("model").and_then(Value::as_str);
        if id.is_some() || model.is_some() {
            normalizer.metadata(
                out,
                ResponseMetadata {
                    id: id.map(str::to_string),
                    model: model.map(str::to_string),
                    request_id: None,
                },
            );
        }
    }

    fn handle_output_item_added(
        &mut self,
        item: Option<&Value>,
        raw: &str,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let Some(item) = item else {
            return;
        };
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
        let item_id = item.get("id").and_then(Value::as_str).unwrap_or("");
        match item_type {
            "function_call" => {
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or(item_id)
                    .to_string();
                self.open_calls.insert(item_id.to_string(), call_id.clone());
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                normalizer.start_tool(
                    out,
                    call_id,
                    name,
                    (!item_id.is_empty()).then(|| item_id.to_string()),
                );
            }
            "reasoning" => {
                normalizer.start_reasoning(out, item_id.to_string());
            }
            "compaction" => {
                normalizer.compaction_start(out);
            }
            other if is_server_tool_item(other) => {
                normalizer.provider_tool_start(out, item_id.to_string(), other.to_string());
            }
            _ => normalizer.raw_frame(out, raw),
        }
    }

    fn handle_content_part_added(
        &mut self,
        item_id: Option<&str>,
        content_index: Option<u64>,
        part: Option<&Value>,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let (Some(item_id), Some(part)) = (item_id, part) else {
            return;
        };
        let part_type = part.get("type").and_then(Value::as_str).unwrap_or("");
        if part_type != "output_text" && part_type != "refusal" {
            return;
        }
        let id = Self::text_block_id(item_id, content_index);
        let metadata = if part_type == "refusal" {
            self.saw_refusal = true;
            ProviderMetadata::with("openai", json!({"content_type": "refusal"}))
        } else {
            ProviderMetadata::default()
        };
        normalizer.start_text_with_metadata(out, id.clone(), metadata);
        self.open_texts
            .entry(item_id.to_string())
            .or_default()
            .push(id);
    }

    fn handle_annotation_added(
        &mut self,
        item_id: Option<&str>,
        content_index: Option<u64>,
        annotation: Option<Value>,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let (Some(item_id), Some(annotation)) = (item_id, annotation) else {
            return;
        };
        let block = Self::text_block_id(item_id, content_index);
        normalizer.citation(
            out,
            Some(block.clone()),
            Citation::from_raw(annotation.clone()),
        );
        self.annotations.entry(block).or_default().push(annotation);
    }

    fn handle_output_item_done(
        &mut self,
        item: Option<&Value>,
        raw: &str,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) -> Result<()> {
        let Some(item) = item else {
            return Ok(());
        };
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
        let item_id = item.get("id").and_then(Value::as_str).unwrap_or("");
        match item_type {
            "function_call" => {
                let final_arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let completed = item
                    .get("status")
                    .and_then(Value::as_str)
                    .is_none_or(|status| status == "completed");
                if let Some(call_id) = self.open_calls.remove(item_id)
                    && completed
                {
                    normalizer.complete_tool(&call_id, final_arguments);
                }
            }
            "reasoning" => {
                let part = decode_output_item(item)?.and_then(|parts| {
                    parts.into_iter().find_map(|part| match part {
                        AssistantPart::Reasoning(reasoning) => Some(reasoning),
                        _ => None,
                    })
                });
                normalizer.end_reasoning(out, item_id, part);
            }
            "message" => {
                // Item metadata arrives after content completion.
                let phase_metadata = item
                    .get("phase")
                    .and_then(Value::as_str)
                    .map(|phase| ProviderMetadata::with("openai", json!({"phase": phase})));
                for id in self.open_texts.remove(item_id).unwrap_or_default() {
                    if let Some(metadata) = &phase_metadata {
                        normalizer.text_metadata(&id, metadata.clone());
                    }
                    if let Some(annotations) = self.annotations.remove(&id) {
                        normalizer.text_metadata(
                            &id,
                            ProviderMetadata::with("openai", json!({"annotations": annotations})),
                        );
                    }
                    normalizer.end_text(out, &id);
                }
            }
            "compaction" => {
                if let Some(AssistantPart::Compaction(part)) =
                    decode_output_item(item)?.and_then(|parts| parts.into_iter().next())
                {
                    normalizer.compaction(out, part);
                }
            }
            other if is_server_tool_item(other) => {
                normalizer.provider_tool_end(
                    out,
                    item_id,
                    ProviderToolPart {
                        id: (!item_id.is_empty()).then(|| item_id.to_string()),
                        kind: other.to_string(),
                        namespace: "openai".into(),
                        payload: item.clone(),
                    },
                );
            }
            _ => normalizer.raw_frame(out, raw),
        }
        Ok(())
    }

    fn handle_event(
        &mut self,
        raw: &str,
        mut envelope: StreamEnvelope,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) -> Result<()> {
        if !self.stamped {
            self.stamped = true;
            normalizer.provider_metadata(out, origin_metadata(self.profile));
        }
        let event_type = std::mem::take(&mut envelope.event_type);
        match event_type.as_str() {
            "response.created" | "response.in_progress" | "response.queued" => {
                Self::handle_response_metadata(envelope.response.as_ref(), normalizer, out);
            }
            "response.output_item.added" => {
                self.handle_output_item_added(envelope.item.as_ref(), raw, normalizer, out);
            }
            "response.content_part.added" => self.handle_content_part_added(
                envelope.item_id.as_deref(),
                envelope.content_index,
                envelope.part.as_ref(),
                normalizer,
                out,
            ),
            "response.output_text.annotation.added" => self.handle_annotation_added(
                envelope.item_id.as_deref(),
                envelope.content_index,
                envelope.annotation,
                normalizer,
                out,
            ),
            "response.output_text.delta" | "response.refusal.delta" => {
                if let (Some(item_id), Some(delta)) = (envelope.item_id.as_deref(), envelope.delta)
                {
                    normalizer.text_delta(
                        out,
                        Self::text_block_id(item_id, envelope.content_index),
                        delta,
                    );
                }
            }
            "response.content_part.done" => {
                // The later item event attaches phase metadata and closes the blocks.
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                if let (Some(item_id), Some(delta)) = (envelope.item_id, envelope.delta) {
                    normalizer.reasoning_delta(out, item_id, delta);
                }
            }
            "response.function_call_arguments.delta" => {
                if let (Some(item_id), Some(delta)) = (envelope.item_id.as_deref(), envelope.delta)
                    && let Some(call_id) = self.open_calls.get(item_id)
                {
                    normalizer.tool_delta(out, call_id, delta);
                }
            }
            "response.output_item.done" => {
                self.handle_output_item_done(envelope.item.as_ref(), raw, normalizer, out)?;
            }
            "response.completed" => {
                self.handle_terminal(envelope.response, "completed", normalizer, out)?;
            }
            "response.incomplete" => {
                self.handle_terminal(envelope.response, "incomplete", normalizer, out)?;
            }
            "response.failed" => {
                self.handle_terminal(envelope.response, "failed", normalizer, out)?;
            }
            "error" => {
                self.errored = true;
                normalizer.error(out, envelope.error.into_error(self.profile));
            }
            other => {
                if let Some(status) = Self::server_tool_status(other)
                    && let Some(item_id) = envelope.item_id
                {
                    normalizer.provider_tool_update(out, item_id, status.to_string());
                }
                normalizer.raw_frame(out, raw);
            }
        }
        Ok(())
    }

    fn handle_terminal(
        &mut self,
        response: Option<Value>,
        status: &str,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) -> Result<()> {
        let parsed: ResponseObject = match response {
            Some(value) => serde_json::from_value(value).map_err(|e| {
                Error::malformed(format!(
                    "{}: invalid terminal response object: {e}",
                    self.profile.as_str()
                ))
            })?,
            None => {
                return Err(Error::malformed(format!(
                    "{}: terminal event carried no response object",
                    self.profile.as_str()
                )));
            }
        };
        if let Some(usage) = &parsed.usage {
            normalizer.merge_usage(&usage.to_usage());
        }
        match status {
            "failed" => {
                if !self.errored {
                    normalizer.error(out, failed_response_error(parsed.error, self.profile));
                }
                normalizer.finish(out, Finish::with_raw(FinishReason::Error, "failed"));
            }
            "incomplete" => {
                let finish =
                    finish_from_status(Some("incomplete"), parsed.incomplete_details.as_ref());
                normalizer.finish(out, finish);
            }
            _ => {
                let finish = if self.saw_refusal {
                    Finish::with_raw(FinishReason::ContentFilter, "refusal")
                } else {
                    finish_from_status(Some("completed"), None)
                };
                normalizer.finish(out, finish);
            }
        }
        Ok(())
    }
}

impl StreamDecoder for ResponsesStreamDecoder {
    fn on_frame(
        &mut self,
        data: String,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) -> Result<()> {
        // Some compatible proxies append a Chat-style [DONE] sentinel.
        if data.trim() == "[DONE]" {
            return Ok(());
        }
        // Type mismatches fail the stream: demoting them would silently lose
        // deltas or the terminal status.
        let envelope: StreamEnvelope = serde_json::from_str(&data).map_err(|e| {
            Error::malformed(format!("{}: invalid stream event: {e}", self.profile))
        })?;
        self.handle_event(&data, envelope, normalizer, out)
    }

    fn on_eof(&mut self, _normalizer: &mut StreamNormalizer, _out: &mut Vec<StreamEvent>) {}
}
