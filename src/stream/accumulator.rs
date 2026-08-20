use std::sync::Arc;

use super::StreamEvent;
use crate::error::{Error, ErrorKind};
use crate::message::AssistantPart;
use crate::metadata::ProviderMetadata;
use crate::response::{Finish, GenerateResult, ResponseMetadata, Warning};
use crate::usage::Usage;

/// Folds [`StreamEvent`]s into a [`GenerateResult`].
///
/// Useful when you want to forward events (e.g. to a UI) while also building
/// the assistant message for history replay:
///
/// ```no_run
/// # use caido_ai::{Result, StreamAccumulator, StreamEvent};
/// # use futures_util::StreamExt;
/// # async fn accumulate(
/// #     mut stream: impl futures_util::Stream<Item = StreamEvent> + Unpin,
/// # ) -> Result<()> {
/// let mut acc = StreamAccumulator::new();
/// while let Some(event) = stream.next().await {
///     acc.push(event);
/// }
/// let result = acc.into_result()?;
/// let history = vec![result.to_assistant_message()];
/// # let _ = history;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    content: Vec<AssistantPart>,
    open_text: Vec<(String, usize)>,
    /// Original content position of each open tool block.
    open_tools: Vec<(String, usize)>,
    warnings: Vec<Warning>,
    metadata: ResponseMetadata,
    finish: Option<Finish>,
    usage: Usage,
    error: Option<Arc<Error>>,
    provider_metadata: ProviderMetadata,
}

impl StreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Place a completed call where its block started, shifting the slots
    /// recorded after that position.
    fn insert_tool_call(&mut self, call: crate::message::ToolCall) {
        let position = self
            .open_tools
            .iter()
            .position(|(open_id, _)| *open_id == call.call_id);
        let index = position
            .map(|position| self.open_tools.remove(position).1)
            .unwrap_or(self.content.len());
        self.content.insert(index, AssistantPart::ToolCall(call));
        for (_, slot) in self.open_text.iter_mut().chain(&mut self.open_tools) {
            if *slot >= index {
                *slot += 1;
            }
        }
    }

    /// Merge end-of-block metadata and drop text blocks that never produced
    /// output, which providers such as Anthropic reject on replay.
    fn close_text(&mut self, index: usize, provider_metadata: ProviderMetadata) {
        let Some(AssistantPart::Text {
            text,
            provider_metadata: existing,
        }) = self.content.get_mut(index)
        else {
            return;
        };
        existing.merge(provider_metadata);
        if !text.is_empty() || !existing.is_empty() {
            return;
        }
        self.content.remove(index);
        for (_, slot) in self.open_text.iter_mut().chain(&mut self.open_tools) {
            if *slot > index {
                *slot -= 1;
            }
        }
    }

    pub fn push(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::StreamStart { warnings } => self.warnings.extend(warnings),
            StreamEvent::ResponseMetadata(metadata) => {
                if metadata.id.is_some() {
                    self.metadata.id = metadata.id;
                }
                if metadata.model.is_some() {
                    self.metadata.model = metadata.model;
                }
                if metadata.request_id.is_some() {
                    self.metadata.request_id = metadata.request_id;
                }
            }
            StreamEvent::TextStart { id, .. } => {
                self.content.push(AssistantPart::Text {
                    text: String::new(),
                    provider_metadata: ProviderMetadata::default(),
                });
                self.open_text.push((id, self.content.len() - 1));
            }
            StreamEvent::TextDelta { id, delta } => {
                let index = self
                    .open_text
                    .iter()
                    .find(|(open_id, _)| *open_id == id)
                    .map(|(_, index)| *index);
                if let Some(index) = index
                    && let Some(AssistantPart::Text { text, .. }) = self.content.get_mut(index)
                {
                    text.push_str(&delta);
                }
            }
            StreamEvent::TextEnd {
                id,
                provider_metadata,
            } => {
                let position = self
                    .open_text
                    .iter()
                    .position(|(open_id, _)| *open_id == id);
                if let Some(position) = position {
                    let (_, index) = self.open_text.remove(position);
                    self.close_text(index, provider_metadata);
                }
            }
            StreamEvent::ReasoningStart { .. } | StreamEvent::ReasoningDelta { .. } => {}
            StreamEvent::ReasoningEnd { part, .. } => {
                self.content.push(AssistantPart::Reasoning(part));
            }
            StreamEvent::ToolInputStart { call_id, .. } => {
                self.open_tools.push((call_id, self.content.len()));
            }
            StreamEvent::ToolInputDelta { .. } | StreamEvent::ToolInputEnd { .. } => {}
            StreamEvent::ToolCall(call) => self.insert_tool_call(call),
            StreamEvent::CompactionStart
            | StreamEvent::CompactionDelta { .. }
            | StreamEvent::CompactionAbort => {}
            StreamEvent::Compaction(part) => {
                self.content.push(AssistantPart::Compaction(part));
            }
            StreamEvent::ProviderToolStart { .. }
            | StreamEvent::ProviderToolUpdate { .. }
            | StreamEvent::ProviderToolAbort { .. } => {}
            StreamEvent::ProviderToolEnd(part) => {
                self.content.push(AssistantPart::ProviderTool {
                    provider_tool: part,
                });
            }
            StreamEvent::Citation { .. } => {}
            StreamEvent::ProviderMetadata(metadata) => {
                self.provider_metadata.merge(metadata);
            }
            StreamEvent::Raw { .. } => {}
            StreamEvent::Error { error } => {
                if self.error.is_none() {
                    self.error = Some(error);
                }
            }
            StreamEvent::Finish { finish, usage } => {
                self.finish = Some(finish);
                self.usage.merge_from(&usage);
            }
        }
    }

    /// Finish accumulation.
    ///
    /// # Errors
    ///
    /// Returns the first stream error that occurred.
    pub fn into_result(self) -> crate::Result<GenerateResult> {
        if let Some(error) = self.error {
            return Err(Arc::try_unwrap(error).unwrap_or_else(|arc| arc.clone_without_source()));
        }
        let finish = self.finish.ok_or_else(|| {
            let mut error = Error::new(
                ErrorKind::TruncatedStream,
                "stream accumulation ended without a finish event",
            );
            if let Some(model) = &self.metadata.model {
                error = error.with_model(model);
            }
            if let Some(request_id) = &self.metadata.request_id {
                error = error.with_request_id(request_id);
            }
            error
        })?;
        Ok(GenerateResult {
            content: self.content,
            finish,
            usage: self.usage,
            warnings: self.warnings,
            response: self.metadata,
            provider_metadata: self.provider_metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::response::FinishReason;

    #[test]
    fn accumulator_builds_result() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.push(StreamEvent::StreamStart { warnings: vec![] });
        accumulator.push(StreamEvent::TextStart {
            id: "0".into(),
            provider_metadata: ProviderMetadata::default(),
        });
        accumulator.push(StreamEvent::TextDelta {
            id: "0".into(),
            delta: "Hello ".into(),
        });
        accumulator.push(StreamEvent::TextDelta {
            id: "0".into(),
            delta: "world".into(),
        });
        accumulator.push(StreamEvent::TextEnd {
            id: "0".into(),
            provider_metadata: ProviderMetadata::default(),
        });
        accumulator.push(StreamEvent::Finish {
            finish: Finish::with_raw(FinishReason::Stop, "stop"),
            usage: Usage {
                output_tokens: Some(2),
                ..Usage::default()
            },
        });
        let result = accumulator.into_result().unwrap();
        assert_eq!(result.text(), "Hello world");
        assert_eq!(result.finish.reason, FinishReason::Stop);
        assert_eq!(result.usage.output_tokens, Some(2));
    }

    #[test]
    fn tool_calls_land_where_their_block_started() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.push(StreamEvent::StreamStart { warnings: vec![] });
        for call_id in ["call_1", "call_2"] {
            accumulator.push(StreamEvent::ToolInputStart {
                call_id: call_id.into(),
                name: "search".into(),
                item_id: None,
                provider_call_id: None,
            });
        }
        accumulator.push(StreamEvent::TextStart {
            id: "0".into(),
            provider_metadata: ProviderMetadata::default(),
        });
        accumulator.push(StreamEvent::TextDelta {
            id: "0".into(),
            delta: "after".into(),
        });
        for call_id in ["call_1", "call_2"] {
            accumulator.push(StreamEvent::ToolInputEnd {
                call_id: call_id.into(),
            });
            accumulator.push(StreamEvent::ToolCall(crate::message::ToolCall {
                call_id: call_id.into(),
                item_id: None,
                provider_call_id: None,
                name: "search".into(),
                arguments: "{}".into(),
                provider_metadata: ProviderMetadata::default(),
            }));
        }
        accumulator.push(StreamEvent::TextDelta {
            id: "0".into(),
            delta: " text".into(),
        });
        accumulator.push(StreamEvent::TextEnd {
            id: "0".into(),
            provider_metadata: ProviderMetadata::default(),
        });
        accumulator.push(StreamEvent::Finish {
            finish: Finish::new(FinishReason::ToolCalls),
            usage: Usage::default(),
        });

        let result = accumulator.into_result().unwrap();
        let kinds: Vec<&str> = result
            .content
            .iter()
            .map(|part| match part {
                AssistantPart::ToolCall(call) => call.call_id.as_str(),
                AssistantPart::Text { text, .. } => text.as_str(),
                _ => "other",
            })
            .collect();
        assert_eq!(kinds, vec!["call_1", "call_2", "after text"]);
    }

    #[test]
    fn empty_text_blocks_without_metadata_are_dropped() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.push(StreamEvent::StreamStart { warnings: vec![] });
        accumulator.push(StreamEvent::TextStart {
            id: "0".into(),
            provider_metadata: ProviderMetadata::default(),
        });
        accumulator.push(StreamEvent::TextEnd {
            id: "0".into(),
            provider_metadata: ProviderMetadata::default(),
        });
        accumulator.push(StreamEvent::TextStart {
            id: "1".into(),
            provider_metadata: ProviderMetadata::default(),
        });
        accumulator.push(StreamEvent::TextDelta {
            id: "1".into(),
            delta: "kept".into(),
        });
        accumulator.push(StreamEvent::TextEnd {
            id: "1".into(),
            provider_metadata: ProviderMetadata::default(),
        });
        accumulator.push(StreamEvent::TextStart {
            id: "2".into(),
            provider_metadata: ProviderMetadata::default(),
        });
        accumulator.push(StreamEvent::TextEnd {
            id: "2".into(),
            provider_metadata: ProviderMetadata::with("gemini", serde_json::json!({"sig": 1})),
        });
        accumulator.push(StreamEvent::Finish {
            finish: Finish::new(FinishReason::Stop),
            usage: Usage::default(),
        });

        let result = accumulator.into_result().unwrap();
        let texts: Vec<&str> = result
            .content
            .iter()
            .map(|part| match part {
                AssistantPart::Text { text, .. } => text.as_str(),
                _ => "other",
            })
            .collect();
        assert_eq!(texts, vec!["kept", ""]);
    }

    #[test]
    fn accumulator_rejects_missing_finish() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.push(StreamEvent::StreamStart { warnings: vec![] });
        accumulator.push(StreamEvent::ResponseMetadata(ResponseMetadata {
            id: Some("resp_56".into()),
            model: Some("gpt-5.6-sol".into()),
            request_id: Some("req_56".into()),
        }));

        let error = accumulator.into_result().unwrap_err();

        assert_eq!(error.kind(), ErrorKind::TruncatedStream);
        assert_eq!(error.model(), Some("gpt-5.6-sol"));
        assert_eq!(error.request_id(), Some("req_56"));
    }
}
