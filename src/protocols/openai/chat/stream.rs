use serde::Deserialize;
use serde_json::{Value, json};

use super::ChatDialect;
use super::types::{
    ChatToolCallFunction, ChatUsage, decode_inline_error, map_chat_finish_reason,
    merge_reasoning_details, reasoning_part_from_details, refusal_metadata,
    result_provider_metadata,
};
use crate::error::{Error, ErrorKind, Result};
use crate::metadata::ProviderMetadata;
use crate::protocols::StreamDecoder;
use crate::response::{Finish, FinishReason, ResponseMetadata};
use crate::stream::{Citation, StreamEvent, StreamNormalizer};

const TEXT_BLOCK: &str = "t0";
const REASONING_BLOCK: &str = "r0";
const REFUSAL_BLOCK: &str = "ref0";

pub(crate) struct ChatStreamDecoder {
    dialect: ChatDialect,
    /// Open tool call ids by index.
    calls_by_index: std::collections::HashMap<u64, String>,
    /// Fragments for calls not yet fully identified (need id + name).
    pending_by_index: std::collections::HashMap<u64, PendingCall>,
    finish: Option<Finish>,
    response_id_sent: bool,
    response_model_sent: bool,
    /// OpenRouter reasoning_details accumulated across deltas.
    reasoning_details: Vec<Value>,
    /// Citations accumulated across deltas and attached as a replace-merged list.
    annotations: Vec<Value>,
    saw_reasoning: bool,
    saw_refusal: bool,
    done: bool,
}

impl ChatStreamDecoder {
    pub(crate) fn new(dialect: ChatDialect) -> Self {
        Self {
            dialect,
            calls_by_index: Default::default(),
            pending_by_index: Default::default(),
            finish: None,
            response_id_sent: false,
            response_model_sent: false,
            reasoning_details: Vec::new(),
            annotations: Vec::new(),
            saw_reasoning: false,
            saw_refusal: false,
            done: false,
        }
    }

    fn close_reasoning(&mut self, normalizer: &mut StreamNormalizer, out: &mut Vec<StreamEvent>) {
        if !self.saw_reasoning {
            return;
        }
        self.saw_reasoning = false;
        let details = merge_reasoning_details(std::mem::take(&mut self.reasoning_details));
        let part = if details.is_empty() {
            None
        } else {
            reasoning_part_from_details(Some(details), None, self.dialect)
        };
        normalizer.end_reasoning(out, REASONING_BLOCK, part);
    }

    /// Emit incomplete calls with synthetic ids instead of dropping them.
    fn flush_pending(&mut self, normalizer: &mut StreamNormalizer, out: &mut Vec<StreamEvent>) {
        let mut indices: Vec<u64> = self.pending_by_index.keys().copied().collect();
        indices.sort_unstable();
        for index in indices {
            let Some(pending) = self.pending_by_index.remove(&index) else {
                continue;
            };
            let Some(name) = pending.name else { continue };
            let call_id = pending.call_id.unwrap_or_else(|| format!("call_{index}"));
            self.calls_by_index.insert(index, call_id.clone());
            normalizer.start_tool(out, call_id.clone(), name, None, None);
            if !pending.arguments.is_empty() {
                normalizer.tool_delta(out, &call_id, pending.arguments);
            }
        }
    }

    /// Complete every open call when the response finishes.
    fn complete_tools(&mut self, normalizer: &mut StreamNormalizer) {
        let mut indices: Vec<u64> = self.calls_by_index.keys().copied().collect();
        indices.sort_unstable();
        for index in indices {
            if let Some(call_id) = self.calls_by_index.remove(&index) {
                normalizer.complete_tool(&call_id, None);
            }
        }
    }

    fn finish_now(&mut self, normalizer: &mut StreamNormalizer, out: &mut Vec<StreamEvent>) {
        if self.done {
            return;
        }
        self.done = true;
        self.close_reasoning(normalizer, out);
        if let Some(finish) = &self.finish
            && finish.reason == FinishReason::Error
        {
            let raw = finish.raw.as_deref().unwrap_or("error");
            normalizer.error(
                out,
                Error::new(
                    ErrorKind::Provider,
                    format!("provider reported terminal finish_reason {raw:?}"),
                )
                .with_origin(self.dialect.profile().as_str()),
            );
            normalizer.finish(out, self.finish.clone().expect("checked above"));
            return;
        }
        let mut finish = self
            .finish
            .clone()
            .unwrap_or_else(|| Finish::new(FinishReason::Stop));
        if self.saw_refusal {
            finish = Finish::with_raw(FinishReason::ContentFilter, "refusal");
        }
        self.flush_pending(normalizer, out);
        self.complete_tools(normalizer);
        normalizer.finish(out, finish);
    }

    fn emit_chunk_metadata(
        &mut self,
        chunk: &ChatChunk,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        normalizer.provider_metadata(
            out,
            result_provider_metadata(chunk.provider.as_ref(), chunk.usage.as_ref(), self.dialect),
        );
        let id = (!self.response_id_sent).then(|| chunk.id.clone()).flatten();
        let model = (!self.response_model_sent)
            .then(|| chunk.model.clone())
            .flatten();
        self.response_id_sent |= id.is_some();
        self.response_model_sent |= model.is_some();
        if id.is_some() || model.is_some() {
            normalizer.metadata(
                out,
                ResponseMetadata {
                    id,
                    model,
                    request_id: None,
                },
            );
        }
        if let Some(usage) = &chunk.usage {
            normalizer.merge_usage(&usage.to_usage(self.dialect.output_token_accounting()));
        }
    }

    fn process_delta(
        &mut self,
        delta: ChunkDelta,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let reasoning_delta = delta.reasoning_content.or(delta.reasoning);
        if let Some(text) = reasoning_delta
            && !text.is_empty()
        {
            self.saw_reasoning = true;
            normalizer.reasoning_delta(out, REASONING_BLOCK.into(), text);
        }
        if let Some(details) = delta.reasoning_details
            && !details.is_empty()
        {
            self.saw_reasoning = true;
            normalizer.start_reasoning(out, REASONING_BLOCK.into());
            self.reasoning_details.extend(details);
        }
        if let Some(text) = delta.content
            && !text.is_empty()
        {
            self.close_reasoning(normalizer, out);
            normalizer.text_delta(out, TEXT_BLOCK.into(), text);
        }
        if let Some(annotations) = delta.annotations
            && !annotations.is_empty()
        {
            for annotation in &annotations {
                normalizer.citation(
                    out,
                    Some(TEXT_BLOCK.into()),
                    Citation::from_raw(annotation.clone()),
                );
            }
            self.annotations.extend(annotations);
            normalizer.text_metadata(
                TEXT_BLOCK,
                ProviderMetadata::with("openai", json!({"annotations": &self.annotations})),
            );
        }
        if let Some(refusal) = delta.refusal
            && !refusal.is_empty()
        {
            self.saw_refusal = true;
            self.close_reasoning(normalizer, out);
            normalizer.start_text_with_metadata(out, REFUSAL_BLOCK.into(), refusal_metadata());
            normalizer.text_delta(out, REFUSAL_BLOCK.into(), refusal);
        }

        let mut tool_calls = delta.tool_calls.unwrap_or_default();
        if tool_calls.is_empty()
            && let Some(function) = delta.function_call
        {
            tool_calls.push(ChunkToolCall {
                index: Some(0),
                id: Some("call_0".to_string()),
                function: Some(function),
            });
        }
        self.process_tool_calls(tool_calls, normalizer, out);
    }

    fn process_tool_calls(
        &mut self,
        tool_calls: Vec<ChunkToolCall>,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        for (position, tool_call) in tool_calls.into_iter().enumerate() {
            self.close_reasoning(normalizer, out);
            self.process_tool_call(position, tool_call, normalizer, out);
        }
    }

    fn process_tool_call(
        &mut self,
        position: usize,
        tool_call: ChunkToolCall,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let index = tool_call.index.unwrap_or(position as u64);
        // Gateways without indices reuse slot zero across distinct calls.
        if let Some(new_id) = tool_call.id.as_deref().filter(|id| !id.is_empty())
            && let Some(open_id) = self.calls_by_index.get(&index)
            && open_id != new_id
        {
            let open_id = open_id.clone();
            self.calls_by_index.remove(&index);
            normalizer.complete_tool(&open_id, None);
        }
        let function = tool_call.function.unwrap_or(ChatToolCallFunction {
            name: None,
            arguments: None,
        });
        if self.calls_by_index.contains_key(&index) {
            if let Some(arguments) = function.arguments
                && let Some(call_id) = self.calls_by_index.get(&index)
                && !arguments.is_empty()
            {
                normalizer.tool_delta(out, call_id, arguments);
            }
            return;
        }

        let pending = self.pending_by_index.entry(index).or_default();
        if let Some(id) = tool_call.id.filter(|id| !id.is_empty()) {
            pending.call_id = Some(id);
        }
        if let Some(name) = function.name.filter(|name| !name.is_empty()) {
            pending.name = Some(name);
        }
        if let Some(arguments) = function.arguments {
            pending.arguments.push_str(&arguments);
        }
        if pending.call_id.is_some() && pending.name.is_some() {
            let pending = self
                .pending_by_index
                .remove(&index)
                .expect("entry just inserted");
            let call_id = pending.call_id.expect("checked");
            let name = pending.name.expect("checked");
            self.calls_by_index.insert(index, call_id.clone());
            normalizer.start_tool(out, call_id.clone(), name, None, None);
            if !pending.arguments.is_empty() {
                normalizer.tool_delta(out, &call_id, pending.arguments);
            }
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChunkToolCall {
    #[serde(default)]
    index: Option<u64>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ChatToolCallFunction>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChunkDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    refusal: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ChunkToolCall>>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    reasoning_details: Option<Vec<Value>>,
    /// Url citations (also OpenRouter web-plugin citations).
    #[serde(default)]
    annotations: Option<Vec<Value>>,
    /// Deprecated single-call streaming form used by some gateways.
    #[serde(default)]
    function_call: Option<ChatToolCallFunction>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChunkChoice {
    #[serde(default)]
    index: Option<u64>,
    #[serde(default)]
    delta: Option<ChunkDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    native_finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatChunk {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
    #[serde(default)]
    error: Option<Value>,
    #[serde(default)]
    provider: Option<Value>,
}

impl StreamDecoder for ChatStreamDecoder {
    fn on_frame(
        &mut self,
        data: String,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) -> Result<()> {
        if self.done {
            return Ok(());
        }
        if data.trim() == "[DONE]" {
            self.finish_now(normalizer, out);
            return Ok(());
        }
        // Type mismatches fail the stream: demoting them would silently lose
        // deltas or the terminal status.
        let chunk: ChatChunk = serde_json::from_str(&data).map_err(|e| {
            Error::malformed(format!(
                "{}: invalid stream chunk: {e}",
                self.dialect.profile()
            ))
        })?;

        // The normalizer aborts partial tool blocks after inline errors.
        if let Some(error) = &chunk.error {
            normalizer.error(out, decode_inline_error(error, self.dialect));
            self.done = true;
            self.close_reasoning(normalizer, out);
            normalizer.finish(out, Finish::with_raw(FinishReason::Error, "error"));
            return Ok(());
        }

        self.emit_chunk_metadata(&chunk, normalizer, out);

        for choice in chunk.choices {
            // Multiple candidates would interleave into one normalized stream.
            if choice.index.unwrap_or(0) != 0 {
                continue;
            }
            if let Some(delta) = choice.delta {
                self.process_delta(delta, normalizer, out);
            }

            if let Some(reason) = choice.finish_reason {
                let raw_reason = choice
                    .native_finish_reason
                    .unwrap_or_else(|| reason.clone());
                self.finish = Some(Finish::with_raw(
                    map_chat_finish_reason(&reason),
                    raw_reason,
                ));
            }
        }
        Ok(())
    }

    fn on_eof(&mut self, normalizer: &mut StreamNormalizer, out: &mut Vec<StreamEvent>) {
        // Compatible servers may omit [DONE] after a finish reason.
        if self.finish.is_some() {
            self.finish_now(normalizer, out);
        }
    }
}

/// A streaming tool call whose identity is still incomplete.
#[derive(Debug, Default)]
pub(crate) struct PendingCall {
    pub(crate) call_id: Option<String>,
    pub(crate) name: Option<String>,
    pub(crate) arguments: String,
}
