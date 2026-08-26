use serde::Deserialize;
use serde_json::{Value, json};

use super::types::{
    AnthropicUsage, MessageObject, anthropic_error_kind, finish_from_stop_reason,
    is_server_tool_block,
};
use crate::error::{Error, ErrorKind, Result};
use crate::message::{CompactionPart, ProviderToolPart, ReasoningContent, ReasoningPart};
use crate::metadata::ProviderMetadata;
use crate::protocols::{ApiProfile, StreamDecoder};
use crate::response::{Finish, FinishReason, ResponseMetadata};
use crate::stream::{Citation, StreamEvent, StreamNormalizer};
use crate::transport::framing::StreamFrame;

/// The block's own id (`server_tool_use`) or the id of the call it answers
/// (`*_tool_result`).
fn provider_tool_block_id(block: &Value) -> Option<String> {
    block
        .get("id")
        .or_else(|| block.get("tool_use_id"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

#[derive(Debug, Default)]
pub(crate) enum BlockKind {
    #[default]
    Unknown,
    Text {
        /// Citations streamed via `citations_delta`, attached to the text
        /// part's metadata when the block closes.
        citations: Vec<Value>,
    },
    Thinking {
        text: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    Tool {
        call_id: String,
        /// Input delivered in content_block_start (complete for deferred calls).
        start_input: Option<String>,
        accumulated: String,
    },
    Compaction {
        content: String,
    },
    /// A provider-executed tool block kept raw for lossless replay.
    ProviderTool {
        block: Value,
        /// Streamed `input_json_delta` fragments, folded into the block's
        /// `input` when it closes.
        input_json: String,
    },
}

pub(crate) struct AnthropicStreamDecoder {
    /// Stamped on stream errors: Anthropic directly or on Bedrock.
    profile: ApiProfile,
    blocks: std::collections::HashMap<u64, BlockKind>,
    finish: Option<Finish>,
    done: bool,
}

impl AnthropicStreamDecoder {
    pub(crate) fn new(profile: ApiProfile) -> Self {
        Self {
            profile,
            blocks: std::collections::HashMap::new(),
            finish: None,
            done: false,
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamEnvelope {
    #[serde(rename = "type", default)]
    event_type: String,
    #[serde(default)]
    index: Option<u64>,
    #[serde(default)]
    message: Option<MessageObject>,
    #[serde(default)]
    content_block: Option<Value>,
    #[serde(default)]
    delta: Option<Value>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
    #[serde(default)]
    error: Option<Value>,
    #[serde(default)]
    context_management: Option<Value>,
}

impl AnthropicStreamDecoder {
    fn block_id(index: u64) -> String {
        index.to_string()
    }

    fn finish_now(&mut self, normalizer: &mut StreamNormalizer, out: &mut Vec<StreamEvent>) {
        if self.done {
            return;
        }
        self.done = true;
        let mut indices: Vec<u64> = self.blocks.keys().copied().collect();
        indices.sort_unstable();
        for index in indices {
            self.close_block(index, false, normalizer, out);
        }
        let finish = self
            .finish
            .clone()
            .unwrap_or_else(|| Finish::new(FinishReason::Stop));
        normalizer.finish(out, finish);
    }

    fn close_block(
        &mut self,
        index: u64,
        completed: bool,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let id = Self::block_id(index);
        match self.blocks.remove(&index) {
            Some(BlockKind::Text { citations }) => {
                if !citations.is_empty() {
                    normalizer.text_metadata(
                        &id,
                        ProviderMetadata::with("anthropic", json!({"citations": citations})),
                    );
                }
                normalizer.end_text(out, &id);
            }
            Some(BlockKind::Thinking { text, signature }) => {
                normalizer.end_reasoning(
                    out,
                    &id,
                    Some(ReasoningPart {
                        id: None,
                        content: vec![ReasoningContent::Text {
                            text,
                            signature: (!signature.is_empty()).then_some(signature),
                        }],
                        provider_metadata: ProviderMetadata::default(),
                    }),
                );
            }
            Some(BlockKind::RedactedThinking { data }) => {
                normalizer.start_reasoning(out, id.clone());
                normalizer.end_reasoning(
                    out,
                    &id,
                    Some(ReasoningPart {
                        id: None,
                        content: vec![ReasoningContent::Redacted { data }],
                        provider_metadata: ProviderMetadata::default(),
                    }),
                );
            }
            Some(BlockKind::Tool {
                call_id,
                start_input,
                accumulated,
            }) => {
                if completed {
                    let final_arguments =
                        if accumulated.trim().is_empty() { start_input } else { Some(accumulated) };
                    normalizer.complete_tool(&call_id, final_arguments);
                }
            }
            Some(BlockKind::Compaction { content }) => {
                if completed {
                    normalizer.compaction(
                        out,
                        CompactionPart {
                            id: None,
                            content: Some(content),
                            encrypted_content: None,
                        },
                    );
                }
            }
            Some(BlockKind::ProviderTool {
                mut block,
                input_json,
            }) => {
                if completed {
                    if !input_json.trim().is_empty()
                        && let Ok(input) = serde_json::from_str::<Value>(&input_json)
                    {
                        block["input"] = input;
                    }
                    let kind = block
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    normalizer.provider_tool_end(
                        out,
                        provider_tool_block_id(&block).as_deref().unwrap_or(&id),
                        ProviderToolPart {
                            id: provider_tool_block_id(&block),
                            kind,
                            namespace: "anthropic".into(),
                            payload: block,
                        },
                    );
                }
            }
            Some(BlockKind::Unknown) | None => {}
        }
    }

    fn handle_message_start(
        envelope: &StreamEnvelope,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let Some(message) = &envelope.message else {
            return;
        };
        normalizer.metadata(
            out,
            ResponseMetadata {
                id: message.id.clone(),
                model: message.model.clone(),
                request_id: None,
            },
        );
        if let Some(usage) = &message.usage {
            normalizer.merge_usage(&usage.to_usage());
        }
    }

    fn start_tool_block(
        &mut self,
        index: u64,
        block: &Value,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let call_id = block
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let name = block
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let start_input = block.get("input").map(Value::to_string);
        self.blocks.insert(
            index,
            BlockKind::Tool {
                call_id: call_id.clone(),
                start_input,
                accumulated: String::new(),
            },
        );
        normalizer.start_tool(out, call_id, name, None, None);
    }

    fn start_other_block(
        &mut self,
        index: u64,
        id: String,
        block: &Value,
        raw: &str,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let block_type = block.get("type").and_then(Value::as_str).unwrap_or("");
        if is_server_tool_block(block_type) {
            self.blocks.insert(
                index,
                BlockKind::ProviderTool {
                    block: block.clone(),
                    input_json: String::new(),
                },
            );
            normalizer.provider_tool_start(
                out,
                provider_tool_block_id(block).unwrap_or(id),
                block_type.to_string(),
            );
        } else {
            self.blocks.insert(index, BlockKind::Unknown);
            normalizer.raw_frame(out, raw);
        }
    }

    fn handle_content_block_start(
        &mut self,
        envelope: &StreamEnvelope,
        raw: &str,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let index = envelope.index.unwrap_or(0);
        let id = Self::block_id(index);
        let Some(block) = &envelope.content_block else {
            return;
        };
        match block.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => {
                self.blocks.insert(
                    index,
                    BlockKind::Text {
                        citations: Vec::new(),
                    },
                );
                normalizer.start_text(out, id);
            }
            "thinking" => {
                self.blocks.insert(
                    index,
                    BlockKind::Thinking {
                        text: String::new(),
                        signature: String::new(),
                    },
                );
                normalizer.start_reasoning(out, id);
            }
            "redacted_thinking" => {
                let data = block
                    .get("data")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                self.blocks
                    .insert(index, BlockKind::RedactedThinking { data });
            }
            "tool_use" => self.start_tool_block(index, block, normalizer, out),
            "compaction" => {
                self.blocks.insert(
                    index,
                    BlockKind::Compaction {
                        content: String::new(),
                    },
                );
                normalizer.compaction_start(out);
            }
            _ => self.start_other_block(index, id, block, raw, normalizer, out),
        }
    }

    fn handle_input_json_delta(
        &mut self,
        index: u64,
        partial: &str,
        raw: &str,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        if partial.is_empty() {
            return;
        }
        match self.blocks.get_mut(&index) {
            Some(BlockKind::Tool {
                call_id,
                accumulated,
                ..
            }) => {
                accumulated.push_str(partial);
                let call_id = call_id.clone();
                normalizer.tool_delta(out, &call_id, partial.to_string());
            }
            Some(BlockKind::ProviderTool { input_json, .. }) => input_json.push_str(partial),
            _ => normalizer.raw_frame(out, raw),
        }
    }

    fn handle_content_block_delta(
        &mut self,
        envelope: &StreamEnvelope,
        raw: &str,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let index = envelope.index.unwrap_or(0);
        let id = Self::block_id(index);
        let Some(delta) = &envelope.delta else {
            return;
        };
        match delta.get("type").and_then(Value::as_str).unwrap_or("") {
            "text_delta" => {
                let text = delta
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                normalizer.text_delta(out, id, text.to_string());
            }
            "thinking_delta" => {
                let text = delta
                    .get("thinking")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(BlockKind::Thinking { text: buffer, .. }) = self.blocks.get_mut(&index)
                {
                    buffer.push_str(text);
                }
                normalizer.reasoning_delta(out, id, text.to_string());
            }
            "signature_delta" => {
                let signature = delta
                    .get("signature")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(BlockKind::Thinking {
                    signature: buffer, ..
                }) = self.blocks.get_mut(&index)
                {
                    buffer.push_str(signature);
                }
            }
            "input_json_delta" => {
                let partial = delta
                    .get("partial_json")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                self.handle_input_json_delta(index, partial, raw, normalizer, out);
            }
            "compaction_delta" => {
                let content = delta
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(BlockKind::Compaction { content: buffer }) = self.blocks.get_mut(&index)
                {
                    buffer.push_str(content);
                    normalizer.compaction_delta(out, content.to_string());
                }
            }
            "citations_delta" => {
                if let Some(citation) = delta.get("citation")
                    && let Some(BlockKind::Text { citations }) = self.blocks.get_mut(&index)
                {
                    citations.push(citation.clone());
                    normalizer.citation(out, Some(id), Citation::from_raw(citation.clone()));
                }
            }
            _ => normalizer.raw_frame(out, raw),
        }
    }

    fn handle_message_delta(
        &mut self,
        envelope: &StreamEnvelope,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        if let Some(context_management) = &envelope.context_management {
            normalizer.provider_metadata(
                out,
                ProviderMetadata::with(
                    "anthropic",
                    json!({"context_management": context_management}),
                ),
            );
        }
        if let Some(usage) = &envelope.usage {
            normalizer.merge_usage(&usage.to_usage());
        }
        if let Some(delta) = &envelope.delta {
            if let Some(stop_details) = delta.get("stop_details") {
                normalizer.provider_metadata(
                    out,
                    ProviderMetadata::with("anthropic", json!({"stop_details": stop_details})),
                );
            }
            if let Some(stop_reason) = delta.get("stop_reason").and_then(Value::as_str) {
                self.finish = Some(finish_from_stop_reason(Some(stop_reason)));
            }
        }
    }

    fn handle_error(
        &mut self,
        envelope: &StreamEnvelope,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let error_type = envelope
            .error
            .as_ref()
            .and_then(|error| error.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let message = envelope
            .error
            .as_ref()
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .unwrap_or("stream error event");
        let kind = anthropic_error_kind(
            (!error_type.is_empty()).then_some(error_type),
            message,
            ErrorKind::Provider,
        );
        let mut error = Error::new(kind, message.to_string()).with_origin(self.profile.as_str());
        if !error_type.is_empty() {
            error = error.with_code(error_type.to_string());
        }
        normalizer.error(out, error);
        self.done = true;
        normalizer.finish(out, Finish::with_raw(FinishReason::Error, "error"));
    }
}

impl StreamDecoder for AnthropicStreamDecoder {
    fn on_frame(
        &mut self,
        frame: StreamFrame,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) -> Result<()> {
        if self.done {
            return Ok(());
        }
        // Type mismatches fail the stream: demoting them would silently lose
        // the stop reason or usage.
        let envelope: StreamEnvelope = serde_json::from_str(&frame.data)
            .map_err(|e| Error::malformed(format!("anthropic: invalid stream event: {e}")))?;
        let raw = frame.data.as_str();

        match envelope.event_type.as_str() {
            "message_start" => {
                Self::handle_message_start(&envelope, normalizer, out);
            }
            "content_block_start" => {
                self.handle_content_block_start(&envelope, raw, normalizer, out);
            }
            "content_block_delta" => {
                self.handle_content_block_delta(&envelope, raw, normalizer, out);
            }
            "content_block_stop" => {
                let index = envelope.index.unwrap_or(0);
                self.close_block(index, true, normalizer, out);
            }
            "message_delta" => {
                self.handle_message_delta(&envelope, normalizer, out);
            }
            "message_stop" => {
                self.finish_now(normalizer, out);
            }
            "ping" => {}
            "error" => {
                self.handle_error(&envelope, normalizer, out);
            }
            _ => normalizer.raw_frame(out, raw),
        }
        Ok(())
    }

    fn on_eof(&mut self, normalizer: &mut StreamNormalizer, out: &mut Vec<StreamEvent>) {
        // Some compatible servers omit message_stop after a stop reason.
        if self.finish.is_some() {
            self.finish_now(normalizer, out);
        }
    }
}
