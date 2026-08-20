//! The refresh state machine behind [`super::OAuthAuthenticator`]: one shared
//! in-flight refresh, tokens persisted before publication, cached failures
//! with a cooldown, and a latched reauthentication state.

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::future::{BoxFuture, FutureExt, Shared};

use crate::auth::{OAuthStatus, TokenStore};
use crate::error::{Error, ErrorKind, Result};

pub(super) type SharedRefresh<T> = Shared<BoxFuture<'static, RefreshOutcome<T>>>;

#[derive(Clone)]
pub(super) enum RefreshOutcome<T> {
    Ready(T),
    RefreshFailed(Arc<Error>),
    PersistenceFailed { tokens: T, error: Arc<Error> },
}

impl<T> RefreshOutcome<T> {
    pub(super) fn error(&self) -> Option<Error> {
        match self {
            Self::Ready(_) => None,
            Self::RefreshFailed(error) | Self::PersistenceFailed { error, .. } => {
                Some(error.clone_without_source())
            }
        }
    }
}

pub(super) fn refresh_operation<T>(
    refresh: impl Future<Output = Result<T>> + Send + 'static,
    store: Option<Arc<dyn TokenStore<T>>>,
) -> SharedRefresh<T>
where
    T: Clone + Send + Sync + 'static,
{
    async move {
        let tokens = match refresh.await {
            Ok(tokens) => tokens,
            Err(error) => return RefreshOutcome::RefreshFailed(Arc::new(error)),
        };
        let Some(store) = store else {
            return RefreshOutcome::Ready(tokens);
        };
        match store.save(&tokens).await {
            Ok(()) => RefreshOutcome::Ready(tokens),
            Err(error) => RefreshOutcome::PersistenceFailed {
                tokens,
                error: Arc::new(error),
            },
        }
    }
    .boxed()
    .shared()
}

pub(super) struct RefreshState<T> {
    pub(super) tokens: T,
    in_flight: Option<(u64, SharedRefresh<T>)>,
    pending_tokens: Option<T>,
    next_operation_id: u64,
    last_completed_operation_id: Option<u64>,
    refresh_backoff_until: Option<Instant>,
    last_refresh_error: Option<Error>,
    reauth_error: Option<Error>,
}

impl<T: Clone> RefreshState<T> {
    /// How long a failed refresh is cached before another attempt.
    const REFRESH_FAILURE_COOLDOWN: Duration = Duration::from_secs(5);

    pub(super) fn new(tokens: T) -> Self {
        Self {
            tokens,
            in_flight: None,
            pending_tokens: None,
            next_operation_id: 0,
            last_completed_operation_id: None,
            refresh_backoff_until: None,
            last_refresh_error: None,
            reauth_error: None,
        }
    }

    pub(super) fn cached_refresh_error(&self, provider: &str) -> Option<Error> {
        if let Some(error) = &self.reauth_error {
            return Some(error.clone_without_source());
        }
        self.refresh_backoff_until
            .filter(|until| Instant::now() < *until)
            .map(|_| {
                self.last_refresh_error
                    .as_ref()
                    .map(Error::clone_without_source)
                    .unwrap_or_else(|| {
                        Error::new(
                            ErrorKind::Authentication,
                            format!("{provider} token refresh recently failed; retry shortly"),
                        )
                    })
            })
    }

    pub(super) fn status(&self) -> OAuthStatus {
        if self.reauth_error.is_some() {
            OAuthStatus::ReauthRequired
        } else if self.in_flight.is_some() {
            OAuthStatus::Refreshing
        } else if self.pending_tokens.is_some() {
            OAuthStatus::PersistenceFailed
        } else if self.last_refresh_error.is_some() {
            OAuthStatus::TransientFailure
        } else {
            OAuthStatus::Ready
        }
    }

    pub(super) fn in_flight(&self) -> Option<(u64, SharedRefresh<T>)> {
        self.in_flight
            .as_ref()
            .map(|(id, operation)| (*id, operation.clone()))
    }

    pub(super) fn pending_tokens(&self) -> Option<T> {
        self.pending_tokens.clone()
    }

    pub(super) fn start(&mut self, operation: SharedRefresh<T>) -> (u64, SharedRefresh<T>) {
        if let Some(existing) = self.in_flight() {
            return existing;
        }
        let id = self.next_operation_id;
        self.next_operation_id = self.next_operation_id.wrapping_add(1);
        self.in_flight = Some((id, operation.clone()));
        (id, operation)
    }

    pub(super) fn complete(&mut self, id: u64, outcome: &RefreshOutcome<T>) -> bool {
        if !matches!(self.in_flight, Some((current, _)) if current == id) {
            return false;
        }
        self.in_flight = None;
        self.last_completed_operation_id = Some(id);
        match outcome {
            RefreshOutcome::Ready(tokens) => {
                self.tokens = tokens.clone();
                self.pending_tokens = None;
                self.refresh_backoff_until = None;
                self.last_refresh_error = None;
                self.reauth_error = None;
            }
            RefreshOutcome::RefreshFailed(error) => {
                self.record_refresh_failure(error);
            }
            RefreshOutcome::PersistenceFailed { tokens, error } => {
                self.pending_tokens = Some(tokens.clone());
                self.refresh_backoff_until = Some(Instant::now() + Self::REFRESH_FAILURE_COOLDOWN);
                self.last_refresh_error = Some(error.clone_without_source());
            }
        }
        true
    }

    pub(super) fn is_latest_completion(&self, id: u64) -> bool {
        self.in_flight.is_none() && self.last_completed_operation_id == Some(id)
    }

    pub(super) fn require_reauthentication(&mut self, provider: &'static str) -> Error {
        let error = Error::new(
            ErrorKind::Authentication,
            format!("{provider} credentials cannot be refreshed; sign in again"),
        )
        .with_origin(provider)
        .with_code("reauth_required");
        self.reauth_error = Some(error.clone_without_source());
        error
    }

    fn record_refresh_failure(&mut self, error: &Error) {
        if refresh_requires_reauthentication(error) {
            self.reauth_error = Some(error.clone_without_source());
            self.refresh_backoff_until = None;
            self.last_refresh_error = None;
        } else {
            self.refresh_backoff_until = Some(Instant::now() + Self::REFRESH_FAILURE_COOLDOWN);
            self.last_refresh_error = Some(error.clone_without_source());
        }
    }
}

/// Error codes that mean the refresh token is dead and the user must sign in again.
pub(super) const DEAD_REFRESH_TOKEN_CODES: &[&str] = &[
    "invalid_grant",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_reused",
    "refresh_token_invalidated",
    "refresh_token_revoked",
];

fn refresh_requires_reauthentication(error: &Error) -> bool {
    error
        .code()
        .is_some_and(|code| DEAD_REFRESH_TOKEN_CODES.contains(&code))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn failed_refresh_is_cached_during_cooldown() {
        let mut state = RefreshState::new(());
        let operation = refresh_operation(
            async { Err(Error::new(ErrorKind::Transport, "refresh failed")) },
            None,
        );
        let (id, future) = state.start(operation);
        let outcome = future.await;
        assert!(state.complete(id, &outcome));

        let cached = state
            .cached_refresh_error("provider")
            .expect("failure should be cached during cooldown");
        assert_eq!(cached.kind(), ErrorKind::Transport);
        assert_eq!(cached.message(), "refresh failed");
    }

    #[tokio::test]
    async fn successful_refresh_publishes_tokens() {
        let mut state = RefreshState::new("old");
        let operation = refresh_operation(async { Ok("new") }, None);
        let (id, future) = state.start(operation);
        let outcome = future.await;
        assert!(state.complete(id, &outcome));

        assert!(state.cached_refresh_error("provider").is_none());
        assert_eq!(state.tokens, "new");
    }

    #[tokio::test]
    async fn invalid_grant_latches_reauthentication_state() {
        let mut state = RefreshState::new(());
        let operation = refresh_operation(
            async {
                Err(Error::new(ErrorKind::Authentication, "dead refresh token")
                    .with_code("invalid_grant"))
            },
            None,
        );
        let (id, future) = state.start(operation);
        let outcome = future.await;
        assert!(state.complete(id, &outcome));

        assert_eq!(state.status(), OAuthStatus::ReauthRequired);
        assert!(state.cached_refresh_error("provider").is_some());
    }

    #[tokio::test]
    async fn refresh_operation_survives_a_cancelled_waiter() {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let operation = refresh_operation(
            async move {
                receiver.await.map_err(|_| {
                    Error::new(ErrorKind::Transport, "refresh result sender was dropped")
                })
            },
            None,
        );
        let mut state = RefreshState::new("old");
        let (_, waiter) = state.start(operation);
        let task = tokio::spawn(waiter);
        tokio::task::yield_now().await;
        task.abort();
        let _ = task.await;

        sender.send("new").expect("refresh operation is retained");
        let (id, retained) = state.in_flight().expect("operation remains in state");
        let outcome = retained.await;
        assert!(state.complete(id, &outcome));

        assert_eq!(state.tokens, "new");
    }
}
