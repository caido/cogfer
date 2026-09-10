//! [`OAuthAuthenticator`] and the traits a refreshable token type implements
//! to use it.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use futures_util::future::BoxFuture;

use super::refresh::{RefreshState, refresh_operation};
use super::{OAuthStatus, expires_within};
use crate::auth::{Rejection, RequestAuthenticator, TokenStore};
use crate::error::Result;
use crate::transport::HttpRequest;

const TARGET: &str = "ai|oauth";

/// A refreshable token set, as seen by [`OAuthAuthenticator`].
///
/// Implement this together with a [`TokenRefresher`] to give any bearer
/// credential with a refresh token (an OAuth subscription, Azure Entra, an
/// STS session) the shared single-flight refresh.
pub trait OAuthTokens: Clone + Send + Sync + 'static {
    fn access_token(&self) -> &str;

    fn refresh_token(&self) -> Option<&str>;

    /// Unix seconds when the access token expires, when known.
    fn expires_at(&self) -> Option<i64>;

    /// Fill in fields derivable from the tokens themselves (JWT claims, for
    /// example) before first use.
    fn normalized(self) -> Self {
        self
    }

    /// Carry forward the fields a refresh response omits.
    fn merge_refreshed(previous: &Self, refreshed: Self) -> Self;

    /// Set the credential headers on an outgoing request.
    ///
    /// # Errors
    ///
    /// Returns an error when a token is not a valid header value.
    fn apply(&self, request: &mut HttpRequest) -> Result<()>;
}

/// The client that exchanges refresh tokens for new token sets.
///
/// It names the provider and owns the refresh policy since the tokens are
/// plain data.
pub trait TokenRefresher<T>: Clone + Send + Sync + fmt::Debug + 'static {
    /// The provider name used in errors and logs.
    const PROVIDER: &'static str;

    /// Refresh this long before expiry so a token cannot lapse mid-request.
    const EXPIRY_SKEW: Duration;

    fn refresh(&self, refresh_token: String) -> BoxFuture<'static, Result<T>>;
}

#[derive(Clone, Copy)]
enum RefreshTrigger<'a> {
    Proactive,
    Unauthorized(Option<&'a str>),
}

/// [`RequestAuthenticator`] for tokens that refresh.
///
/// Concurrent requests share one refresh. A configured [`TokenStore`] saves
/// rotated tokens before they become visible to requests. Refreshes run only
/// while a request awaits them: a cancelled waiter parks the in-flight refresh
/// until the next request resumes it. A proactive refresh that fails while the
/// current token is still valid falls back to that token. A 401 rejecting the
/// current token triggers one refresh and retry. Other statuses do not.
#[must_use = "authenticator modifiers return an updated value"]
pub struct OAuthAuthenticator<T, R> {
    refresher: R,
    /// An async lock because every accessor is `async`. It is `futures_util`'s
    /// rather than tokio's since tokio is an optional transport dependency.
    /// The guard is released before awaiting an in-flight refresh.
    state: futures_util::lock::Mutex<RefreshState<T>>,
    token_store: Option<Arc<dyn TokenStore<T>>>,
}

impl<T: OAuthTokens, R: TokenRefresher<T>> fmt::Debug for OAuthAuthenticator<T, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OAuthAuthenticator")
            .field("provider", &R::PROVIDER)
            .field("refresher", &self.refresher)
            .field("token_store", &self.token_store.is_some())
            .finish_non_exhaustive()
    }
}

impl<T: OAuthTokens, R: TokenRefresher<T>> OAuthAuthenticator<T, R> {
    /// Create an authenticator from `tokens` and the client that issued them.
    pub fn new(tokens: T, refresher: R) -> Self {
        Self {
            refresher,
            state: futures_util::lock::Mutex::new(RefreshState::new(tokens.normalized())),
            token_store: None,
        }
    }

    /// Durably save refreshed tokens before publishing them to requests.
    pub fn with_token_store(mut self, store: Arc<dyn TokenStore<T>>) -> Self {
        self.token_store = Some(store);
        self
    }

    /// The current published token set.
    pub async fn tokens(&self) -> T {
        self.state.lock().await.tokens.clone()
    }

    /// Current credential status.
    pub async fn status(&self) -> OAuthStatus {
        let mut state = self.state.lock().await;
        if state.status() == OAuthStatus::Ready
            && state.tokens.refresh_token().is_none()
            && expires_within(state.tokens.expires_at(), Duration::ZERO)
        {
            state.require_reauthentication(R::PROVIDER);
        }
        state.status()
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
                        expires_within(state.tokens.expires_at(), R::EXPIRY_SKEW),
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
                    if let Some(error) = state.cached_refresh_error(R::PROVIDER) {
                        if usable {
                            return Ok(state.tokens.clone());
                        }
                        return Err(error);
                    }

                    if let Some(tokens) = pending {
                        let Some(store) = self.token_store.clone() else {
                            return Err(state.require_reauthentication(R::PROVIDER));
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
                        log::debug!(target: TARGET, "{} access token requires refresh", R::PROVIDER);
                        state.start(operation)
                    } else if usable {
                        return Ok(state.tokens.clone());
                    } else {
                        return Err(state.require_reauthentication(R::PROVIDER));
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
                            R::PROVIDER
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

#[async_trait::async_trait]
impl<T: OAuthTokens, R: TokenRefresher<T>> RequestAuthenticator for OAuthAuthenticator<T, R> {
    async fn authenticate(&self, request: &mut HttpRequest) -> Result<()> {
        self.apply(request, RefreshTrigger::Proactive).await
    }

    async fn reauthenticate(
        &self,
        request: &mut HttpRequest,
        rejection: &Rejection<'_>,
    ) -> Result<bool> {
        // Only an unauthorized response means the token itself was rejected.
        if rejection.status != 401 {
            return Ok(false);
        }
        let rejected = request.bearer_token().map(str::to_owned);
        self.apply(request, RefreshTrigger::Unauthorized(rejected.as_deref()))
            .await?;
        Ok(true)
    }
}
