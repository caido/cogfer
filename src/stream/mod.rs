//! Normalized streaming: events, the terminal contract, and accumulation.
//!
//! ## The terminal contract
//!
//! Once [`crate::LanguageModel::stream`] returns an [`EventStream`], the
//! stream guarantees:
//!
//! 1. The first event is [`StreamEvent::StreamStart`].
//! 2. Every `TextStart`/`ReasoningStart`/`ToolInputStart` is closed by its
//!    matching end event before the stream finishes, synthesized if the
//!    provider never sent one. Open provider-tool and compaction lifecycles
//!    similarly close with their completed part or an explicit abort event.
//! 3. Exactly one [`StreamEvent::Finish`] is emitted, always last. A failed
//!    or truncated stream emits [`StreamEvent::Error`] followed by a `Finish`
//!    with [`crate::FinishReason::Error`], even if the provider claims a normal
//!    completion after the error.
//! 4. After `Finish`, the stream is fused and yields `None`.
//! 5. Open tool blocks close with `ToolInputEnd` only. A [`StreamEvent::ToolCall`]
//!    is emitted only after the provider authoritatively completes the call
//!    with valid JSON arguments.
//!
//! Failures *before* the stream is established (connection refused, auth
//! rejected, request validation) are returned as `Err` from `stream()` itself.

mod accumulator;
mod event;
mod event_stream;
mod normalizer;

pub use self::accumulator::StreamAccumulator;
pub use self::event::{Citation, StreamEvent};
pub use self::event_stream::EventStream;
pub(crate) use self::normalizer::StreamNormalizer;
