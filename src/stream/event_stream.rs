use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::stream::BoxStream;
use futures_util::{Stream, StreamExt};

use super::{StreamAccumulator, StreamEvent};
use crate::response::GenerateResult;

/// The normalized event stream returned by [`crate::LanguageModel::stream`].
pub struct EventStream {
    inner: futures_util::stream::Fuse<BoxStream<'static, StreamEvent>>,
}

impl std::fmt::Debug for EventStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventStream").finish_non_exhaustive()
    }
}

impl EventStream {
    pub(crate) fn new(inner: BoxStream<'static, StreamEvent>) -> Self {
        Self {
            inner: inner.fuse(),
        }
    }

    /// Drain the stream and assemble the final [`GenerateResult`].
    ///
    /// # Errors
    ///
    /// Returns the first stream [`crate::Error`] that occurred.
    pub async fn collect_result(mut self) -> crate::Result<GenerateResult> {
        let mut accumulator = StreamAccumulator::new();
        while let Some(event) = self.inner.next().await {
            accumulator.push(event);
        }
        accumulator.into_result()
    }
}

impl Stream for EventStream {
    type Item = StreamEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.poll_next_unpin(cx)
    }
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;

    use super::*;
    use crate::response::{Finish, FinishReason};
    use crate::usage::Usage;

    #[tokio::test]
    async fn event_stream_is_fused_after_exhaustion() {
        let events = vec![
            StreamEvent::StreamStart { warnings: vec![] },
            StreamEvent::Finish {
                finish: Finish::new(FinishReason::Stop),
                usage: Usage::default(),
            },
        ];
        let inner = futures_util::stream::unfold(events.into_iter(), |mut events| async move {
            events.next().map(|event| (event, events))
        });
        let mut stream = EventStream::new(Box::pin(inner));
        assert!(stream.next().await.is_some());
        assert!(stream.next().await.is_some());
        assert!(stream.next().await.is_none());
        assert!(stream.next().await.is_none());
    }
}
