//! Splitting a streamed response body into the frames a protocol decodes.
//!
//! Providers stream either Server-Sent Events or, on AWS, the binary event
//! stream encoding. Both deliver a sequence of self-contained payloads, and a
//! [`FrameSource`] turns raw body chunks into those payloads so the protocol
//! decoders never see the framing.

/// One payload from a streamed response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StreamFrame {
    /// The frame payload, normally one JSON document.
    pub(crate) data: String,
    /// Set when the framing layer itself carried an error (an AWS event
    /// stream exception, for example). `data` then holds the error payload.
    pub(crate) exception: Option<String>,
}

impl StreamFrame {
    pub(crate) fn data(data: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            exception: None,
        }
    }
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
