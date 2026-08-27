use std::sync::Arc;

use serde_json::Value;

use crate::error::Error;
use crate::message::{ReasoningPart, ToolCall};
use crate::metadata::ProviderMetadata;
use crate::response::{Finish, ResponseMetadata, Warning};
use crate::usage::Usage;

/// A source citation / grounding reference attached to streamed text.
///
/// Common citation fields plus the exact provider JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Citation {
    pub url: Option<String>,
    pub title: Option<String>,
    /// The quoted source passage, when the provider includes one.
    pub cited_text: Option<String>,
    /// The provider's exact citation/annotation JSON.
    pub raw: Value,
}

impl Citation {
    /// Best-effort extraction from a provider citation object: the common
    /// fields are probed under their known provider spellings, and the full
    /// object is kept as `raw`.
    pub(crate) fn from_raw(raw: Value) -> Self {
        let field = |names: &[&str]| {
            names
                .iter()
                .find_map(|name| raw.get(name).and_then(Value::as_str))
                .map(str::to_string)
        };
        Citation {
            url: field(&["url", "uri"]),
            title: field(&["title"]),
            cited_text: field(&["cited_text", "snippet", "quote"]),
            raw,
        }
    }
}

/// A normalized stream event.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum StreamEvent {
    /// Always first. Carries the warnings raised while lowering the request.
    StreamStart {
        warnings: Vec<Warning>,
    },
    /// Response identity, as soon as the provider reveals it.
    ResponseMetadata(ResponseMetadata),
    TextStart {
        id: String,
        /// Metadata known at start and merged into the block's `TextEnd` event.
        provider_metadata: ProviderMetadata,
    },
    TextDelta {
        id: String,
        delta: String,
    },
    TextEnd {
        id: String,
        /// Provider extras for the whole text block (e.g. Gemini thought
        /// signatures attached to text parts).
        provider_metadata: ProviderMetadata,
    },
    ReasoningStart {
        id: String,
    },
    /// Incremental human-visible reasoning text.
    ReasoningDelta {
        id: String,
        delta: String,
    },
    /// Carries the complete typed reasoning part (signatures, encrypted and
    /// redacted content included) so history replay needs nothing else.
    ReasoningEnd {
        id: String,
        part: ReasoningPart,
    },
    ToolInputStart {
        call_id: String,
        name: String,
        item_id: Option<String>,
    },
    /// Incremental raw-JSON argument text.
    ToolInputDelta {
        call_id: String,
        delta: String,
    },
    ToolInputEnd {
        call_id: String,
    },
    /// The complete tool call, emitted immediately after its `ToolInputEnd`.
    ToolCall(ToolCall),
    /// The provider began compacting context. Frontends can surface progress
    /// during what would otherwise be a long silent interval.
    CompactionStart,
    /// Display-only compaction summary text before the authoritative part.
    /// Anthropic streams this text. OpenAI compaction is opaque.
    CompactionDelta {
        delta: String,
    },
    /// The in-flight compaction was abandoned before the provider supplied an
    /// authoritative replayable part.
    CompactionAbort,
    /// A provider-produced compaction item for same-profile replay.
    Compaction(crate::message::CompactionPart),
    /// A provider-executed tool started. `kind` is its item or block type.
    ProviderToolStart {
        id: String,
        kind: String,
    },
    /// A provider-named status change for an in-flight server tool.
    ProviderToolUpdate {
        id: String,
        status: String,
    },
    /// The provider tool stopped without an authoritative completed item.
    ProviderToolAbort {
        id: String,
    },
    /// The finished server-side tool item, exactly as the provider reported
    /// it. Also lands in the result content for same-profile replay.
    ProviderToolEnd(crate::message::ProviderToolPart),
    /// A source citation / grounding reference. `text_id` names the text
    /// block it annotates when the provider ties citations to one.
    Citation {
        text_id: Option<String>,
        citation: Citation,
    },
    /// Response-level namespaced provider extras discovered mid-stream
    /// (e.g. Anthropic applied context edits). Merged into
    /// [`crate::GenerateResult::provider_metadata`] by the accumulator.
    ProviderMetadata(ProviderMetadata),
    /// An unrecognized provider payload, only when
    /// [`crate::Request::include_raw_events`] is set.
    Raw {
        value: Value,
    },
    /// A stream failure followed by an error `Finish`.
    Error {
        error: Arc<Error>,
    },
    /// Always last, exactly once.
    Finish {
        finish: Finish,
        usage: Usage,
    },
}
