use std::collections::HashSet;

use futures_util::StreamExt;
use cogfer::{EventStream, FinishReason, GenerateResult, StreamAccumulator, StreamEvent};

pub(crate) async fn drain(stream: EventStream) -> Vec<StreamEvent> {
    stream.collect::<Vec<_>>().await
}

/// Assemble the [`GenerateResult`] a consumer would build from `events`.
pub(crate) fn collect(events: &[StreamEvent]) -> cogfer::Result<GenerateResult> {
    let mut accumulator = StreamAccumulator::new();
    for event in events {
        accumulator.push(event.clone());
    }
    accumulator.into_result()
}

/// Compact event-kind labels for order assertions.
pub(crate) fn kinds(events: &[StreamEvent]) -> Vec<&'static str> {
    events
        .iter()
        .map(|event| match event {
            StreamEvent::StreamStart { .. } => "stream-start",
            StreamEvent::ResponseMetadata(_) => "response-metadata",
            StreamEvent::TextStart { .. } => "text-start",
            StreamEvent::TextDelta { .. } => "text-delta",
            StreamEvent::TextEnd { .. } => "text-end",
            StreamEvent::ReasoningStart { .. } => "reasoning-start",
            StreamEvent::ReasoningDelta { .. } => "reasoning-delta",
            StreamEvent::ReasoningEnd { .. } => "reasoning-end",
            StreamEvent::ToolInputStart { .. } => "tool-input-start",
            StreamEvent::ToolInputDelta { .. } => "tool-input-delta",
            StreamEvent::ToolInputEnd { .. } => "tool-input-end",
            StreamEvent::ToolCall(_) => "tool-call",
            StreamEvent::CompactionStart => "compaction-start",
            StreamEvent::CompactionDelta { .. } => "compaction-delta",
            StreamEvent::CompactionAbort => "compaction-abort",
            StreamEvent::Compaction(_) => "compaction",
            StreamEvent::ProviderToolStart { .. } => "provider-tool-start",
            StreamEvent::ProviderToolUpdate { .. } => "provider-tool-update",
            StreamEvent::ProviderToolAbort { .. } => "provider-tool-abort",
            StreamEvent::ProviderToolEnd(_) => "provider-tool-end",
            StreamEvent::Citation { .. } => "citation",
            StreamEvent::ProviderMetadata(_) => "provider-metadata",
            StreamEvent::Raw { .. } => "raw",
            StreamEvent::Error { .. } => "error",
            StreamEvent::Finish { .. } => "finish",
            _ => "other",
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum OpenBlock<'a> {
    Text(&'a str),
    Reasoning(&'a str),
    ToolInput(&'a str),
    ProviderTool { id: &'a str, kind: &'a str },
}

#[derive(Default)]
struct StreamContract<'a> {
    seen: HashSet<OpenBlock<'a>>,
    open: HashSet<OpenBlock<'a>>,
    closed_tool_inputs: HashSet<&'a str>,
    emitted_tool_calls: HashSet<&'a str>,
    compaction_open: bool,
    error_seen: bool,
    finish_seen: bool,
}

impl<'a> StreamContract<'a> {
    fn start(&mut self, block: OpenBlock<'a>) {
        match block {
            OpenBlock::Text(id) | OpenBlock::Reasoning(id) | OpenBlock::ToolInput(id) => {
                assert!(!id.is_empty(), "block id must not be empty");
            }
            OpenBlock::ProviderTool { id, kind } => {
                assert!(!id.is_empty(), "provider tool id must not be empty");
                assert!(!kind.is_empty(), "provider tool kind must not be empty");
            }
        }
        assert!(self.seen.insert(block), "duplicate block {block:?}");
        assert!(self.open.insert(block));
    }

    fn delta(&self, block: OpenBlock<'a>) {
        assert!(
            self.open.contains(&block),
            "delta for non-open block {block:?}"
        );
    }

    fn end(&mut self, block: OpenBlock<'a>) {
        assert!(self.open.remove(&block), "end for non-open block {block:?}");
    }

    fn provider_tool_end(&mut self, id: Option<&'a str>, kind: &'a str) {
        let block = id.map_or_else(
            || {
                let mut matches = self.open.iter().copied().filter(|block| {
                    matches!(block, OpenBlock::ProviderTool { kind: open, .. } if *open == kind)
                });
                let matching = matches.next();
                assert!(
                    matches.next().is_none(),
                    "provider tool end without id is ambiguous for kind {kind:?}"
                );
                matching.unwrap_or_else(|| panic!("provider tool end without start: {kind:?}"))
            },
            |id| OpenBlock::ProviderTool { id, kind },
        );
        self.end(block);
    }

    fn provider_tool_abort(&mut self, id: &'a str) {
        let mut matches = self.open.iter().copied().filter(
            |block| matches!(block, OpenBlock::ProviderTool { id: open, .. } if *open == id),
        );
        let matching = matches
            .next()
            .unwrap_or_else(|| panic!("provider tool abort without start: {id:?}"));
        assert!(
            matches.next().is_none(),
            "provider tool abort is ambiguous for id {id:?}"
        );
        self.end(matching);
    }

    fn tool_call(&mut self, events: &[StreamEvent], index: usize, call_id: &'a str) {
        assert!(
            !self.error_seen,
            "tool call {call_id:?} emitted after Error"
        );
        assert!(
            self.closed_tool_inputs.remove(call_id),
            "tool call {call_id:?} has no matching ToolInputEnd"
        );
        assert!(
            self.emitted_tool_calls.insert(call_id),
            "duplicate tool call {call_id:?}"
        );
        assert!(
            matches!(
                index.checked_sub(1).and_then(|previous| events.get(previous)),
                Some(StreamEvent::ToolInputEnd { call_id: ended }) if ended == call_id
            ),
            "tool call {call_id:?} must immediately follow its ToolInputEnd"
        );
    }

    fn assert_allowed_after_error(&self, event: &StreamEvent) {
        if self.error_seen {
            assert!(
                matches!(
                    event,
                    StreamEvent::TextEnd { .. }
                        | StreamEvent::ReasoningEnd { .. }
                        | StreamEvent::ToolInputEnd { .. }
                        | StreamEvent::ProviderToolAbort { .. }
                        | StreamEvent::ProviderToolEnd(_)
                        | StreamEvent::CompactionAbort
                        | StreamEvent::Compaction(_)
                        | StreamEvent::Finish { .. }
                ),
                "non-terminal event emitted after Error"
            );
        }
    }

    fn assert_complete(&self) {
        assert!(self.finish_seen, "missing Finish event");
        assert!(self.open.is_empty(), "unclosed blocks: {:?}", self.open);
        assert!(!self.compaction_open, "unclosed compaction lifecycle");
    }
}

/// Assert the standard terminal contract over a drained event list.
pub(crate) fn assert_terminal_contract(events: &[StreamEvent]) {
    assert!(
        matches!(events.first(), Some(StreamEvent::StreamStart { .. })),
        "first event must be StreamStart, got {:?}",
        kinds(events).first()
    );
    let mut contract = StreamContract::default();
    for (index, event) in events.iter().enumerate() {
        assert!(
            !contract.finish_seen,
            "event emitted after Finish at index {index}"
        );
        contract.assert_allowed_after_error(event);
        match event {
            StreamEvent::StreamStart { .. } => {
                assert_eq!(index, 0, "StreamStart must occur exactly once");
            }
            StreamEvent::TextStart { id, .. } => contract.start(OpenBlock::Text(id)),
            StreamEvent::TextDelta { id, .. } => contract.delta(OpenBlock::Text(id)),
            StreamEvent::TextEnd { id, .. } => contract.end(OpenBlock::Text(id)),
            StreamEvent::ReasoningStart { id } => contract.start(OpenBlock::Reasoning(id)),
            StreamEvent::ReasoningDelta { id, .. } => contract.delta(OpenBlock::Reasoning(id)),
            StreamEvent::ReasoningEnd { id, .. } => contract.end(OpenBlock::Reasoning(id)),
            StreamEvent::ToolInputStart { call_id, .. } => {
                contract.start(OpenBlock::ToolInput(call_id));
            }
            StreamEvent::ToolInputDelta { call_id, .. } => {
                contract.delta(OpenBlock::ToolInput(call_id));
            }
            StreamEvent::ToolInputEnd { call_id } => {
                contract.end(OpenBlock::ToolInput(call_id));
                assert!(contract.closed_tool_inputs.insert(call_id));
            }
            StreamEvent::ToolCall(call) => {
                assert!(
                    call.parse_arguments::<serde_json::Value>().is_ok(),
                    "tool call {:?} has invalid JSON arguments",
                    call.call_id
                );
                contract.tool_call(events, index, &call.call_id);
            }
            StreamEvent::ProviderToolStart { id, kind } => {
                contract.start(OpenBlock::ProviderTool { id, kind });
            }
            StreamEvent::ProviderToolUpdate { id, .. } => assert!(
                contract.open.iter().any(
                    |block| matches!(block, OpenBlock::ProviderTool { id: open, .. } if *open == id.as_str())
                ),
                "provider tool update for non-open id {id:?}"
            ),
            StreamEvent::ProviderToolEnd(part) => {
                contract.provider_tool_end(part.id.as_deref(), &part.kind);
            }
            StreamEvent::ProviderToolAbort { id } => contract.provider_tool_abort(id),
            StreamEvent::CompactionStart => {
                assert!(
                    !contract.compaction_open,
                    "overlapping compaction lifecycles"
                );
                contract.compaction_open = true;
            }
            StreamEvent::CompactionDelta { .. } => assert!(
                contract.compaction_open,
                "compaction delta without CompactionStart"
            ),
            StreamEvent::CompactionAbort | StreamEvent::Compaction(_) => {
                assert!(
                    contract.compaction_open,
                    "compaction terminal event without CompactionStart"
                );
                contract.compaction_open = false;
            }
            StreamEvent::Citation {
                text_id: Some(text_id),
                ..
            } => assert!(
                contract.seen.contains(&OpenBlock::Text(text_id.as_str())),
                "citation references unknown text block {text_id:?}"
            ),
            StreamEvent::Error { .. } => {
                assert!(!contract.error_seen, "stream emitted more than one Error");
                contract.error_seen = true;
            }
            StreamEvent::Finish { finish, .. } => {
                assert_eq!(
                    finish.reason == FinishReason::Error,
                    contract.error_seen,
                    "Error and error Finish must occur together"
                );
                contract.finish_seen = true;
            }
            _ => {}
        }
    }
    contract.assert_complete();
}
