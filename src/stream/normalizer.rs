use serde_json::Value;

use super::{Citation, StreamEvent};
use crate::error::Error;
#[cfg(test)]
use crate::error::ErrorKind;
use crate::message::{AssistantPart, ReasoningContent, ReasoningPart, ToolCall};
use crate::metadata::ProviderMetadata;
use crate::response::{Finish, FinishReason, GenerateResult, ResponseMetadata};
use crate::usage::Usage;

const TARGET: &str = "ai|stream";

/// Enforces block lifecycles and the terminal contract for every decoder.
#[derive(Debug)]
pub(crate) struct StreamNormalizer {
    open: Vec<OpenBlock>,
    usage: Usage,
    finished: bool,
    errored: bool,
    error_context: Option<ErrorContext>,
    include_raw: bool,
}

#[derive(Debug)]
struct ErrorContext {
    origin: String,
    model: String,
    request_id: Option<String>,
}

#[derive(Debug)]
enum OpenBlock {
    Text {
        id: String,
        provider_metadata: ProviderMetadata,
    },
    Reasoning {
        id: String,
        buffer: String,
    },
    Tool {
        call_id: String,
        name: String,
        item_id: Option<String>,
        arguments: String,
        provider_metadata: ProviderMetadata,
        completed: bool,
        /// Complete arguments that override accumulated deltas.
        final_arguments: Option<String>,
    },
    ProviderTool {
        id: String,
    },
    Compaction,
}

impl StreamNormalizer {
    pub(crate) fn new(include_raw: bool) -> Self {
        Self {
            open: Vec::new(),
            usage: Usage::default(),
            finished: false,
            errored: false,
            error_context: None,
            include_raw,
        }
    }

    pub(crate) fn with_error_context(
        mut self,
        origin: impl Into<String>,
        model: impl Into<String>,
        request_id: Option<String>,
    ) -> Self {
        self.error_context = Some(ErrorContext {
            origin: origin.into(),
            model: model.into(),
            request_id,
        });
        self
    }

    fn enrich_error(&self, mut error: Error) -> Error {
        let Some(context) = &self.error_context else {
            return error;
        };
        if error.origin().is_none() {
            error = error.with_origin(&context.origin);
        }
        if error.model().is_none() {
            error = error.with_model(&context.model);
        }
        if error.request_id().is_none()
            && let Some(request_id) = &context.request_id
        {
            error = error.with_request_id(request_id);
        }
        error
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.finished
    }

    pub(crate) fn metadata(&mut self, out: &mut Vec<StreamEvent>, metadata: ResponseMetadata) {
        out.push(StreamEvent::ResponseMetadata(metadata));
    }

    pub(crate) fn start_text(&mut self, out: &mut Vec<StreamEvent>, id: String) {
        self.start_text_with_metadata(out, id, ProviderMetadata::default());
    }

    /// Open a text block with metadata available from its start.
    pub(crate) fn start_text_with_metadata(
        &mut self,
        out: &mut Vec<StreamEvent>,
        id: String,
        metadata: ProviderMetadata,
    ) {
        if !self.text_is_open(&id) {
            self.open.push(OpenBlock::Text {
                id: id.clone(),
                provider_metadata: metadata.clone(),
            });
            out.push(StreamEvent::TextStart {
                id,
                provider_metadata: metadata,
            });
        }
    }

    pub(crate) fn text_delta(&mut self, out: &mut Vec<StreamEvent>, id: String, delta: String) {
        self.start_text(out, id.clone());
        out.push(StreamEvent::TextDelta { id, delta });
    }

    /// Attach provider metadata for delivery on the block's `TextEnd` event.
    pub(crate) fn text_metadata(&mut self, id: &str, metadata: ProviderMetadata) {
        if let Some(OpenBlock::Text {
            provider_metadata, ..
        }) = self
            .open
            .iter_mut()
            .find(|block| matches!(block, OpenBlock::Text { id: open, .. } if open == id))
        {
            provider_metadata.merge(metadata);
        }
    }

    pub(crate) fn end_text(&mut self, out: &mut Vec<StreamEvent>, id: &str) {
        let position = self
            .open
            .iter()
            .position(|block| matches!(block, OpenBlock::Text { id: open, .. } if open == id));
        if let Some(position) = position {
            let OpenBlock::Text {
                provider_metadata, ..
            } = self.open.remove(position)
            else {
                unreachable!("position points at a text block");
            };
            out.push(StreamEvent::TextEnd {
                id: id.to_string(),
                provider_metadata,
            });
        }
    }

    pub(crate) fn start_reasoning(&mut self, out: &mut Vec<StreamEvent>, id: String) {
        if !self.reasoning_is_open(&id) {
            self.open.push(OpenBlock::Reasoning {
                id: id.clone(),
                buffer: String::new(),
            });
            out.push(StreamEvent::ReasoningStart { id });
        }
    }

    pub(crate) fn reasoning_delta(
        &mut self,
        out: &mut Vec<StreamEvent>,
        id: String,
        delta: String,
    ) {
        self.start_reasoning(out, id.clone());
        if let Some(OpenBlock::Reasoning { buffer, .. }) = self
            .open
            .iter_mut()
            .find(|block| matches!(block, OpenBlock::Reasoning { id: open, .. } if *open == id))
        {
            buffer.push_str(&delta);
        }
        out.push(StreamEvent::ReasoningDelta { id, delta });
    }

    /// Close a reasoning block using typed content or accumulated delta text.
    pub(crate) fn end_reasoning(
        &mut self,
        out: &mut Vec<StreamEvent>,
        id: &str,
        part: Option<ReasoningPart>,
    ) {
        let position = self
            .open
            .iter()
            .position(|block| matches!(block, OpenBlock::Reasoning { id: open, .. } if open == id));
        let Some(position) = position else {
            // Synthesize a start event to keep the lifecycle balanced.
            if part.is_some() {
                self.start_reasoning(out, id.to_string());
                return self.end_reasoning(out, id, part);
            }
            return;
        };
        let OpenBlock::Reasoning { buffer, .. } = self.open.remove(position) else {
            unreachable!("position points at a reasoning block");
        };
        let mut part = part.unwrap_or_else(|| ReasoningPart {
            id: None,
            content: Vec::new(),
            provider_metadata: ProviderMetadata::default(),
        });
        // Preserve visible deltas when typed content is empty.
        if part.content.is_empty() && !buffer.is_empty() {
            part.content.push(ReasoningContent::Text {
                text: buffer,
                signature: None,
            });
        }
        out.push(StreamEvent::ReasoningEnd {
            id: id.to_string(),
            part,
        });
    }

    pub(crate) fn start_tool(
        &mut self,
        out: &mut Vec<StreamEvent>,
        call_id: String,
        name: String,
        item_id: Option<String>,
    ) {
        if !self.tool_is_open(&call_id) {
            self.open.push(OpenBlock::Tool {
                call_id: call_id.clone(),
                name: name.clone(),
                item_id: item_id.clone(),
                arguments: String::new(),
                provider_metadata: ProviderMetadata::default(),
                completed: false,
                final_arguments: None,
            });
            out.push(StreamEvent::ToolInputStart {
                call_id,
                name,
                item_id,
            });
        }
    }

    pub(crate) fn tool_delta(&mut self, out: &mut Vec<StreamEvent>, call_id: &str, delta: String) {
        let Some(OpenBlock::Tool { arguments, .. }) = self.open.iter_mut().find(
            |block| matches!(block, OpenBlock::Tool { call_id: open, .. } if open == call_id),
        ) else {
            // A tool block cannot be synthesized without its name.
            return;
        };
        arguments.push_str(&delta);
        out.push(StreamEvent::ToolInputDelta {
            call_id: call_id.to_string(),
            delta,
        });
    }

    /// Attach provider metadata to the synthesized [`ToolCall`].
    pub(crate) fn tool_metadata(&mut self, call_id: &str, metadata: ProviderMetadata) {
        if let Some(OpenBlock::Tool {
            provider_metadata, ..
        }) = self
            .open
            .iter_mut()
            .find(|block| matches!(block, OpenBlock::Tool { call_id: open, .. } if open == call_id))
        {
            provider_metadata.merge(metadata);
        }
    }

    /// Mark a tool complete and preserve any authoritative final arguments.
    ///
    /// The block remains open until [`StreamNormalizer::finish`] applies the
    /// single cross-protocol promotion policy.
    pub(crate) fn complete_tool(&mut self, call_id: &str, final_arguments: Option<String>) {
        if let Some(OpenBlock::Tool {
            completed,
            final_arguments: slot,
            ..
        }) = self
            .open
            .iter_mut()
            .find(|block| matches!(block, OpenBlock::Tool { call_id: open, .. } if open == call_id))
        {
            *completed = true;
            if final_arguments.is_some() {
                *slot = final_arguments;
            }
        }
    }

    /// Emit completed tools in opening order and report invalid arguments.
    fn emit_completed_tools(&mut self, out: &mut Vec<StreamEvent>) -> bool {
        let mut emitted = false;
        let mut index = 0;
        while index < self.open.len() {
            if !matches!(
                self.open[index],
                OpenBlock::Tool {
                    completed: true,
                    ..
                }
            ) {
                index += 1;
                continue;
            }
            let OpenBlock::Tool {
                call_id,
                name,
                item_id,
                arguments,
                provider_metadata,
                final_arguments,
                ..
            } = self.open.remove(index)
            else {
                unreachable!("index points at a tool block");
            };
            out.push(StreamEvent::ToolInputEnd {
                call_id: call_id.clone(),
            });
            if self.errored {
                continue;
            }
            let mut call = ToolCall {
                call_id,
                item_id,
                name,
                arguments: final_arguments.unwrap_or(arguments),
                provider_metadata,
            };
            call.normalize_blank_arguments();
            if let Err(source) = call.arguments_value() {
                self.error(
                    out,
                    Error::malformed(format!(
                        "tool call `{}` completed with invalid JSON arguments: {source}",
                        call.call_id
                    ))
                    .with_source(source),
                );
                continue;
            }
            out.push(StreamEvent::ToolCall(call));
            emitted = true;
        }
        emitted
    }

    pub(crate) fn compaction_start(&mut self, out: &mut Vec<StreamEvent>) {
        if !self.compaction_is_open() {
            self.open.push(OpenBlock::Compaction);
            out.push(StreamEvent::CompactionStart);
        }
    }

    /// Only Anthropic streams human-readable compaction summary text.
    pub(crate) fn compaction_delta(&mut self, out: &mut Vec<StreamEvent>, delta: String) {
        if !self.compaction_is_open() {
            self.compaction_start(out);
        }
        out.push(StreamEvent::CompactionDelta { delta });
    }

    pub(crate) fn provider_tool_start(
        &mut self,
        out: &mut Vec<StreamEvent>,
        id: String,
        kind: String,
    ) {
        if !self.provider_tool_is_open(&id) {
            self.open.push(OpenBlock::ProviderTool { id: id.clone() });
            out.push(StreamEvent::ProviderToolStart { id, kind });
        }
    }

    pub(crate) fn provider_tool_update(
        &mut self,
        out: &mut Vec<StreamEvent>,
        id: String,
        status: String,
    ) {
        out.push(StreamEvent::ProviderToolUpdate { id, status });
    }

    pub(crate) fn provider_tool_end(
        &mut self,
        out: &mut Vec<StreamEvent>,
        id: &str,
        part: crate::message::ProviderToolPart,
    ) {
        let position = self
            .open
            .iter()
            .position(|block| matches!(block, OpenBlock::ProviderTool { id: open } if open == id));
        if let Some(position) = position {
            self.open.remove(position);
        } else {
            // Synthesize missing starts to keep the lifecycle balanced.
            self.provider_tool_start(out, id.to_string(), part.kind.clone());
            let position = self.open.iter().position(
                |block| matches!(block, OpenBlock::ProviderTool { id: open } if open == id),
            );
            if let Some(position) = position {
                self.open.remove(position);
            }
        }
        out.push(StreamEvent::ProviderToolEnd(part));
    }

    pub(crate) fn citation(
        &mut self,
        out: &mut Vec<StreamEvent>,
        text_id: Option<String>,
        citation: Citation,
    ) {
        out.push(StreamEvent::Citation { text_id, citation });
    }

    /// Close an open compaction block that produced nothing (e.g. a refusal
    /// during the compaction pass) without emitting an empty part.
    pub(crate) fn compaction_abort(&mut self, out: &mut Vec<StreamEvent>) {
        let position = self
            .open
            .iter()
            .position(|block| matches!(block, OpenBlock::Compaction));
        if let Some(position) = position {
            self.open.remove(position);
            out.push(StreamEvent::CompactionAbort);
        }
    }

    pub(crate) fn compaction(
        &mut self,
        out: &mut Vec<StreamEvent>,
        part: crate::message::CompactionPart,
    ) {
        let position = self
            .open
            .iter()
            .position(|block| matches!(block, OpenBlock::Compaction));
        if let Some(position) = position {
            self.open.remove(position);
        } else {
            // Synthesize a display start for an authoritative completed item.
            self.compaction_start(out);
            let position = self
                .open
                .iter()
                .position(|block| matches!(block, OpenBlock::Compaction));
            if let Some(position) = position {
                self.open.remove(position);
            }
        }
        out.push(StreamEvent::Compaction(part));
    }

    /// Response-level extras, dropped when the decoder found none.
    pub(crate) fn provider_metadata(
        &mut self,
        out: &mut Vec<StreamEvent>,
        metadata: ProviderMetadata,
    ) {
        if !metadata.is_empty() {
            out.push(StreamEvent::ProviderMetadata(metadata));
        }
    }

    /// Emit an unrecognized provider frame, parsing it only when raw events
    /// were requested.
    pub(crate) fn raw_frame(&mut self, out: &mut Vec<StreamEvent>, data: &str) {
        if self.include_raw
            && let Ok(value) = serde_json::from_str::<Value>(data)
        {
            out.push(StreamEvent::Raw { value });
        }
    }

    /// Replay a buffered result through the regular block lifecycle, so a
    /// provider that answered a streaming request with one JSON body still
    /// honours the stream contract.
    pub(crate) fn replay(&mut self, out: &mut Vec<StreamEvent>, result: GenerateResult) {
        if result.response.id.is_some() || result.response.model.is_some() {
            self.metadata(
                out,
                ResponseMetadata {
                    id: result.response.id,
                    model: result.response.model,
                    request_id: None,
                },
            );
        }
        self.provider_metadata(out, result.provider_metadata);
        for (index, part) in result.content.into_iter().enumerate() {
            let id = index.to_string();
            match part {
                AssistantPart::Text {
                    text,
                    provider_metadata,
                } => {
                    self.start_text_with_metadata(out, id.clone(), provider_metadata);
                    if !text.is_empty() {
                        self.text_delta(out, id.clone(), text);
                    }
                    self.end_text(out, &id);
                }
                AssistantPart::Reasoning(part) => {
                    self.start_reasoning(out, id.clone());
                    let visible = part.visible_text();
                    if !visible.is_empty() {
                        self.reasoning_delta(out, id.clone(), visible);
                    }
                    self.end_reasoning(out, &id, Some(part));
                }
                AssistantPart::ToolCall(call) => {
                    self.start_tool(out, call.call_id.clone(), call.name, call.item_id);
                    self.tool_metadata(&call.call_id, call.provider_metadata);
                    self.tool_delta(out, &call.call_id, call.arguments);
                    self.complete_tool(&call.call_id, None);
                }
                AssistantPart::Compaction(part) => self.compaction(out, part),
                AssistantPart::ProviderTool { provider_tool } => {
                    let id = provider_tool.id.clone().unwrap_or(id);
                    self.provider_tool_end(out, &id, provider_tool);
                }
            }
        }
        self.merge_usage(&result.usage);
        self.finish(out, result.finish);
    }

    pub(crate) fn merge_usage(&mut self, usage: &Usage) {
        self.usage.merge_from(usage);
    }

    pub(crate) fn error(&mut self, out: &mut Vec<StreamEvent>, error: Error) {
        if self.errored {
            return;
        }
        let error = self.enrich_error(error);
        log::debug!(target: TARGET, "stream error event: {error}");
        self.errored = true;
        out.push(StreamEvent::Error { error });
    }

    pub(crate) fn fail(&mut self, out: &mut Vec<StreamEvent>, error: Error) {
        self.error(out, error);
        self.finish(out, Finish::new(FinishReason::Error));
    }

    /// Close open blocks, promote valid tools, and emit the terminal event.
    ///
    /// Completed calls are promoted unless the finish reason discards them. A
    /// provider `Stop` carrying calls becomes [`FinishReason::ToolCalls`], so
    /// every decoder exposes the same tool lifecycle.
    pub(crate) fn finish(&mut self, out: &mut Vec<StreamEvent>, mut finish: Finish) {
        if self.finished {
            return;
        }
        if !self.errored && !finish.reason.discards_tool_calls() {
            let emitted = self.emit_completed_tools(out);
            if emitted && finish.reason == FinishReason::Stop {
                finish.reason = FinishReason::ToolCalls;
            }
        }
        if self.errored && finish.reason != FinishReason::Error {
            finish.reason = FinishReason::Error;
        }
        self.close_all_open(out);
        self.finished = true;
        out.push(StreamEvent::Finish {
            finish,
            usage: std::mem::take(&mut self.usage),
        });
    }

    pub(crate) fn on_eof(&mut self, out: &mut Vec<StreamEvent>, error: Error) {
        if self.finished {
            return;
        }
        let error = self.enrich_error(error);
        log::warn!(target: TARGET, "stream truncated before a terminal event: {error}");
        self.close_all_open(out);
        self.fail(out, error);
    }

    fn close_all_open(&mut self, out: &mut Vec<StreamEvent>) {
        while let Some(block) = self.open.pop() {
            match block {
                OpenBlock::Text {
                    id,
                    provider_metadata,
                } => {
                    out.push(StreamEvent::TextEnd {
                        id,
                        provider_metadata,
                    });
                }
                OpenBlock::Reasoning { id, buffer } => {
                    let mut part = ReasoningPart {
                        id: None,
                        content: Vec::new(),
                        provider_metadata: ProviderMetadata::default(),
                    };
                    if !buffer.is_empty() {
                        part.content.push(ReasoningContent::Text {
                            text: buffer,
                            signature: None,
                        });
                    }
                    out.push(StreamEvent::ReasoningEnd { id, part });
                }
                OpenBlock::Tool { call_id, .. } => {
                    out.push(StreamEvent::ToolInputEnd { call_id });
                }
                OpenBlock::ProviderTool { id } => {
                    out.push(StreamEvent::ProviderToolAbort { id });
                }
                OpenBlock::Compaction => {
                    out.push(StreamEvent::CompactionAbort);
                }
            }
        }
    }

    fn text_is_open(&self, id: &str) -> bool {
        self.open
            .iter()
            .any(|block| matches!(block, OpenBlock::Text { id: open, .. } if open == id))
    }

    fn reasoning_is_open(&self, id: &str) -> bool {
        self.open
            .iter()
            .any(|block| matches!(block, OpenBlock::Reasoning { id: open, .. } if open == id))
    }

    fn tool_is_open(&self, call_id: &str) -> bool {
        self.open
            .iter()
            .any(|block| matches!(block, OpenBlock::Tool { call_id: open, .. } if open == call_id))
    }

    fn provider_tool_is_open(&self, id: &str) -> bool {
        self.open
            .iter()
            .any(|block| matches!(block, OpenBlock::ProviderTool { id: open } if open == id))
    }

    fn compaction_is_open(&self) -> bool {
        self.open
            .iter()
            .any(|block| matches!(block, OpenBlock::Compaction))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizer_closes_uncompleted_tools_without_promoting_them() {
        let mut normalizer = StreamNormalizer::new(false);
        let mut out = Vec::new();
        normalizer.text_delta(&mut out, "t0".into(), "hello".into());
        normalizer.start_tool(&mut out, "call_1".into(), "search".into(), None);
        normalizer.tool_delta(&mut out, "call_1", "{\"q\":1}".into());
        normalizer.finish(&mut out, Finish::new(FinishReason::Stop));

        let kinds: Vec<&'static str> = out
            .iter()
            .map(|event| match event {
                StreamEvent::TextStart { .. } => "text-start",
                StreamEvent::TextDelta { .. } => "text-delta",
                StreamEvent::TextEnd { .. } => "text-end",
                StreamEvent::ToolInputStart { .. } => "tool-input-start",
                StreamEvent::ToolInputDelta { .. } => "tool-input-delta",
                StreamEvent::ToolInputEnd { .. } => "tool-input-end",
                StreamEvent::ToolCall(_) => "tool-call",
                StreamEvent::Finish { .. } => "finish",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            vec![
                "text-start",
                "text-delta",
                "tool-input-start",
                "tool-input-delta",
                "tool-input-end",
                "text-end",
                "finish",
            ]
        );
    }

    #[test]
    fn truncated_tool_block_never_becomes_a_tool_call() {
        let mut normalizer = StreamNormalizer::new(false);
        let mut out = Vec::new();
        normalizer.start_tool(&mut out, "call_1".into(), "delete_all".into(), None);
        normalizer.tool_delta(&mut out, "call_1", "{\"path\":\"/ho".into());
        normalizer.on_eof(
            &mut out,
            Error::new(
                ErrorKind::TruncatedStream,
                "stream ended before a terminal event was received",
            )
            .with_origin("openai"),
        );
        assert!(
            out.iter()
                .any(|event| matches!(event, StreamEvent::ToolInputEnd { .. }))
        );
        assert!(
            !out.iter()
                .any(|event| matches!(event, StreamEvent::ToolCall(_)))
        );
        assert!(matches!(out[out.len() - 2], StreamEvent::Error { .. }));
    }

    #[test]
    fn eof_without_finish_is_truncation() {
        let mut normalizer = StreamNormalizer::new(false);
        let mut out = Vec::new();
        normalizer.text_delta(&mut out, "t0".into(), "partial".into());
        normalizer.on_eof(
            &mut out,
            Error::new(
                ErrorKind::TruncatedStream,
                "stream ended before a terminal event was received",
            )
            .with_origin("openai"),
        );
        assert!(matches!(out[out.len() - 2], StreamEvent::Error { .. }));
        let Some(StreamEvent::Finish { finish, .. }) = out.last() else {
            panic!("missing finish");
        };
        assert_eq!(finish.reason, FinishReason::Error);
        assert!(
            out.iter()
                .any(|event| matches!(event, StreamEvent::TextEnd { .. }))
        );
    }

    #[test]
    fn finish_after_error_is_forced_to_error_reason() {
        let mut normalizer = StreamNormalizer::new(false);
        let mut out = Vec::new();
        normalizer.text_delta(&mut out, "t0".into(), "partial".into());
        normalizer.error(&mut out, Error::malformed("mid-stream failure"));
        normalizer.finish(&mut out, Finish::with_raw(FinishReason::Stop, "stop"));
        let Some(StreamEvent::Finish { finish, .. }) = out.last() else {
            panic!("missing finish");
        };
        assert_eq!(finish.reason, FinishReason::Error);
        assert_eq!(finish.raw.as_deref(), Some("stop"));
    }

    #[test]
    fn length_finish_aborts_open_tool_blocks() {
        let mut normalizer = StreamNormalizer::new(false);
        let mut out = Vec::new();
        normalizer.start_tool(&mut out, "call_1".into(), "search".into(), None);
        normalizer.tool_delta(&mut out, "call_1", "{\"q\":\"trunc".into());
        normalizer.finish(&mut out, Finish::new(FinishReason::Length));
        assert!(
            out.iter()
                .any(|event| matches!(event, StreamEvent::ToolInputEnd { .. }))
        );
        assert!(
            !out.iter()
                .any(|event| matches!(event, StreamEvent::ToolCall(_)))
        );
        let Some(StreamEvent::Finish { finish, .. }) = out.last() else {
            panic!("missing finish");
        };
        assert_eq!(finish.reason, FinishReason::Length);
    }

    #[test]
    fn completed_tools_are_promoted_even_when_the_provider_says_stop() {
        let mut normalizer = StreamNormalizer::new(false);
        let mut out = Vec::new();
        normalizer.start_tool(&mut out, "call_1".into(), "search".into(), None);
        normalizer.tool_delta(&mut out, "call_1", "{\"q\":1}".into());
        normalizer.complete_tool("call_1", None);
        normalizer.finish(&mut out, Finish::with_raw(FinishReason::Stop, "stop"));

        let position = out
            .iter()
            .position(
                |event| matches!(event, StreamEvent::ToolCall(call) if call.call_id == "call_1"),
            )
            .expect("completed call is emitted");
        assert!(matches!(
            out[position - 1],
            StreamEvent::ToolInputEnd { ref call_id } if call_id == "call_1"
        ));
        let Some(StreamEvent::Finish { finish, .. }) = out.last() else {
            panic!("missing finish");
        };
        assert_eq!(finish.reason, FinishReason::ToolCalls);
        assert_eq!(finish.raw.as_deref(), Some("stop"));
    }

    #[test]
    fn completed_tools_are_not_promoted_when_output_was_truncated() {
        let mut normalizer = StreamNormalizer::new(false);
        let mut out = Vec::new();
        normalizer.start_tool(&mut out, "call_1".into(), "search".into(), None);
        normalizer.tool_delta(&mut out, "call_1", "{\"q\":1}".into());
        normalizer.complete_tool("call_1", None);
        normalizer.finish(&mut out, Finish::with_raw(FinishReason::Length, "length"));

        assert!(
            out.iter()
                .any(|event| matches!(event, StreamEvent::ToolInputEnd { .. }))
        );
        assert!(
            !out.iter()
                .any(|event| matches!(event, StreamEvent::ToolCall(_)))
        );
        let Some(StreamEvent::Finish { finish, .. }) = out.last() else {
            panic!("missing finish");
        };
        assert_eq!(finish.reason, FinishReason::Length);
    }

    #[test]
    fn final_arguments_override_accumulated_deltas() {
        let mut normalizer = StreamNormalizer::new(false);
        let mut out = Vec::new();
        normalizer.start_tool(&mut out, "call_1".into(), "search".into(), None);
        normalizer.tool_delta(&mut out, "call_1", "{\"q\":".into());
        normalizer.complete_tool("call_1", Some("{\"q\":1}".into()));
        normalizer.finish(&mut out, Finish::new(FinishReason::ToolCalls));

        let call = out
            .iter()
            .find_map(|event| match event {
                StreamEvent::ToolCall(call) => Some(call),
                _ => None,
            })
            .expect("tool call is emitted");
        assert_eq!(call.arguments, "{\"q\":1}");
    }

    #[test]
    fn tool_calls_after_an_invalid_call_are_not_emitted() {
        let mut normalizer = StreamNormalizer::new(false);
        let mut out = Vec::new();
        for (id, arguments) in [("invalid", "{"), ("valid", "{}")] {
            normalizer.start_tool(&mut out, id.into(), "tool".into(), None);
            normalizer.tool_delta(&mut out, id, arguments.into());
        }

        normalizer.complete_tool("invalid", None);
        normalizer.complete_tool("valid", None);
        normalizer.finish(&mut out, Finish::new(FinishReason::ToolCalls));

        assert!(
            !out.iter()
                .any(|event| matches!(event, StreamEvent::ToolCall(_)))
        );
        let Some(StreamEvent::Finish { finish, .. }) = out.last() else {
            panic!("missing finish");
        };
        assert_eq!(finish.reason, FinishReason::Error);
    }
}
