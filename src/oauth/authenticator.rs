//! [`OAuthAuthenticator`] and the traits provider token types implement for it.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use futures_util::future::BoxFuture;

use super::OAuthStatus;
use super::expires_within;
use super::refresh::{RefreshState, refresh_operation};
use crate::auth::TokenStore;
use crate::error::Result;
use crate::transport::HttpRequest;

const TARGET: &str = "ai|oauth";

/// A refreshable OAuth token set, as seen by [`OAuthAuthenticator`].
pub(crate) trait OAuthTokens: Clone + Send + Sync + 'static {
    fn access_token(&self) -> &str;

    fn refresh_token(&self) -> Option<&str>;

    /// Unix seconds when the access token expires, when known.
    fn expires_at(&self) -> Option<i64>;

    /// Carry forward the fields a refresh response omits.
    fn merge_refreshed(previous: &Self, refreshed: Self) -> Self;

    /// Set the credential headers on an outgoing request.
    ///
    /// # Errors
    ///
    /// Returns an error when a token is not a valid header value.
    fn apply(&self, request: &mut HttpRequest) -> Result<()>;
}

/// The provider client that exchanges refresh tokens.
pub(crate) trait TokenRefresher<T>: Clone + Send + Sync + fmt::Debug + 'static {
    fn refresh(&self, refresh_token: String) -> BoxFuture<'static, Result<T>>;
}

#[derive(Clone, Copy)]
enum RefreshTrigger<'a> {
    Proactive,
    Unauthorized(Option<&'a str>),
}

/// The refresh policy every provider authenticator shares.
///
/// Concurrent requests share one refresh. A configured [`TokenStore`] saves
/// rotated tokens before they become visible to requests. Refreshes run only
/// while a request awaits them: a cancelled waiter parks the in-flight refresh
/// until the next request resumes it. A proactive refresh that fails while the
/// current token is still valid falls back to that token.
pub(crate) struct OAuthAuthenticator<T, R> {
    provider: &'static str,
    /// Refresh this long before expiry so a token cannot lapse mid-request.
    expiry_skew: Duration,
    refresher: R,
    /// An async lock because every accessor is `async`. It is `futures_util`'s
    /// rather than tokio's since tokio is an optional transport dependency.
    /// The guard is released before awaiting an in-flight refresh.
    state: futures_util::lock::Mutex<RefreshState<T>>,
    token_store: Option<Arc<dyn TokenStore<T>>>,
}

impl<T, R: fmt::Debug> fmt::Debug for OAuthAuthenticator<T, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OAuthAuthenticator")
            .field("provider", &self.provider)
            .field("oauth", &self.refresher)
            .field("token_store", &self.token_store.is_some())
            .finish_non_exhaustive()
    }
}

impl<T: OAuthTokens, R: TokenRefresher<T>> OAuthAuthenticator<T, R> {
    pub(crate) fn new(
        provider: &'static str,
        expiry_skew: Duration,
        tokens: T,
        refresher: R,
    ) -> Self {
        Self {
            provider,
            expiry_skew,
            refresher,
            state: futures_util::lock::Mutex::new(RefreshState::new(tokens)),
            token_store: None,
        }
    }

    pub(crate) fn with_token_store(mut self, store: Arc<dyn TokenStore<T>>) -> Self {
        self.token_store = Some(store);
        self
    }

    /// The current published token set.
    pub(crate) async fn tokens(&self) -> T {
        self.state.lock().await.tokens.clone()
    }

    /// Current OAuth credential status.
    pub(crate) async fn status(&self) -> OAuthStatus {
        let mut state = self.state.lock().await;
        if state.status() == OAuthStatus::Ready
            && state.tokens.refresh_token().is_none()
            && expires_within(state.tokens.expires_at(), Duration::ZERO)
        {
            state.require_reauthentication(self.provider);
        }
        state.status()
    }

    pub(crate) async fn authenticate(&self, request: &mut HttpRequest) -> Result<()> {
        self.apply(request, RefreshTrigger::Proactive).await
    }

    pub(crate) async fn reauthenticate(
        &self,
        request: &mut HttpRequest,
        status: u16,
    ) -> Result<bool> {
        // Only an unauthorized response means the token itself was rejected.
        if status != 401 {
            return Ok(false);
        }
        let rejected = request.bearer_token().map(str::to_owned);
        self.apply(request, RefreshTrigger::Unauthorized(rejected.as_deref()))
            .await?;
        Ok(true)
    }

    async fn apply(&self, request: &mut HttpRequest, trigger: RefreshTrigger<'_>) -> Result<()> {
        self.tokens_for_request(trigger).await?.apply(request)?;
        Ok(())
    }

    async fn tokens_for_request(&self, trigger: RefreshTrigger<'_>) -> Result<T> {
        loop {
            let (usable, operation) = {
                let mut state = self.state.lock().await;
                let expired = expires_within(state.tokens.expires_at(), Duration::ZERO);
                let (rejected_current, needs_refresh) = match trigger {
                    RefreshTrigger::Proactive => (
                        false,
                        expires_within(state.tokens.expires_at(), self.expiry_skew),
                    ),
                    RefreshTrigger::Unauthorized(rejected) => {
                        let current =
                            rejected.is_none_or(|token| token == state.tokens.access_token());
                        (current, current || expired)
                    }
                };
                // Whether the current token can still be sent if refreshing fails.
                let usable = !expired && !rejected_current;

                let operation = if let Some(operation) = state.in_flight() {
                    operation
                } else {
                    let pending = state.pending_tokens();
                    if pending.is_none() && !needs_refresh {
                        return Ok(state.tokens.clone());
                    }
                    if let Some(error) = state.cached_refresh_error(self.provider) {
                        if usable {
                            return Ok(state.tokens.clone());
                        }
                        return Err(error);
                    }

                    if let Some(tokens) = pending {
                        let Some(store) = self.token_store.clone() else {
                            return Err(state.require_reauthentication(self.provider));
                        };
                        let operation = refresh_operation(async move { Ok(tokens) }, Some(store));
                        state.start(operation)
                    } else if let Some(refresh_token) = state.tokens.refresh_token() {
                        let previous = state.tokens.clone();
                        let refresh = self.refresher.refresh(refresh_token.to_owned());
                        let operation = refresh_operation(
                            async move {
                                let refreshed = refresh.await?;
                                Ok(T::merge_refreshed(&previous, refreshed))
                            },
                            self.token_store.clone(),
                        );
                        log::debug!(target: TARGET, "{} access token requires refresh", self.provider);
                        state.start(operation)
                    } else if usable {
                        return Ok(state.tokens.clone());
                    } else {
                        return Err(state.require_reauthentication(self.provider));
                    }
                };
                (usable, operation)
            };

            let (id, future) = operation;
            let outcome = future.await;
            let error = outcome.error();
            let mut state = self.state.lock().await;
            let completed = state.complete(id, &outcome);
            if completed || state.is_latest_completion(id) {
                return match error {
                    Some(error) if usable => {
                        log::warn!(
                            target: TARGET,
                            "{} token refresh failed; using the current token until it expires: {error}",
                            self.provider
                        );
                        Ok(state.tokens.clone())
                    }
                    Some(error) => Err(error),
                    None => Ok(state.tokens.clone()),
                };
            }
        }
    }
}
