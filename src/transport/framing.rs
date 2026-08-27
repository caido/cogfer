//! Splitting a streamed response body into the frames a protocol decodes.
//!
//! Providers stream either Server-Sent Events or, on AWS, the binary event
//! stream encoding. Both deliver a sequence of self-contained payloads, and a
//! [`FrameSource`] turns raw body chunks into those payloads so the protocol
//! decoders never see the framing.

/// One frame from a streamed response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StreamFrame {
    /// A payload for the protocol decoder, normally one JSON document.
    Data(String),
    /// An error the framing layer itself carried (an AWS event stream
    /// exception, for example), named by `kind` with its error payload.
    Exception { kind: String, payload: String },
}

/// Incremental framing of a streamed body.
///
/// Corruption (invalid UTF-8, a frame over the size limit, a failed
/// checksum) is not an item: the source stops producing frames, and the
/// runner reads [`FrameSource::corruption`] to end the stream with an error
/// after delivering the frames that preceded it.
pub(crate) trait FrameSource: Send {
    /// Push bytes and return every frame they complete.
    fn push(&mut self, bytes: &[u8]) -> Vec<StreamFrame>;

    /// Flush a final frame at end of stream, where the framing allows one.
    fn finish(&mut self) -> Option<StreamFrame>;

    /// The reason framing stopped, once it has.
    fn corruption(&self) -> Option<&'static str>;
}
