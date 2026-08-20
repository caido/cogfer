//! Structured errors for every failure class the SDK distinguishes.

use std::fmt;
use std::time::Duration;

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// The class of an [`Error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Invalid client or provider configuration.
    Configuration,
    /// The request itself is invalid and was rejected before any network I/O.
    InvalidRequest,
    /// The selected API profile cannot represent a requested operation.
    UnsupportedCapability,
    /// Request or history content the selected API profile cannot safely represent.
    UnsupportedContent,
    /// Credentials missing, invalid or expired.
    Authentication,
    /// Authenticated but not allowed (permissions, billing or entitlement).
    Permission,
    /// Provider safety, moderation or content-policy enforcement blocked output.
    ContentPolicy,
    /// Model or resource not found.
    NotFound,
    /// Provider rate limit hit.
    RateLimited,
    /// Provider reports overload / temporary unavailability.
    Overloaded,
    /// Prompt does not fit the model context window.
    ContextLength,
    /// Network / transport level failure.
    Transport,
    /// The request or stream timed out.
    Timeout,
    /// The provider returned data the protocol decoder could not understand.
    MalformedResponse,
    /// The stream ended before a terminal event was received.
    TruncatedStream,
    /// Any other upstream provider failure.
    Provider,
}

/// Structured SDK error.
///
/// Carries a stable [`ErrorKind`], a credential-free message, and optional
/// provider diagnostics. The SDK does not generally retry generation, though
/// a refreshable authenticator may recover once from a pre-output 401. Use
/// [`Error::retryable`] and [`Error::retry_after`] to implement host policy.
pub struct Error {
    inner: Box<ErrorInner>,
}

struct ErrorInner {
    kind: ErrorKind,
    message: String,
    origin: Option<String>,
    model: Option<String>,
    status: Option<u16>,
    code: Option<String>,
    request_id: Option<String>,
    retry_after: Option<Duration>,
    retryable: Option<bool>,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl Error {
    /// Create a new error of `kind` with a human-readable, credential-free message.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            inner: Box::new(ErrorInner {
                kind,
                message: message.into(),
                origin: None,
                model: None,
                status: None,
                code: None,
                request_id: None,
                retry_after: None,
                retryable: None,
                source: None,
            }),
        }
    }

    /// Shorthand for a [`ErrorKind::Configuration`] error.
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Configuration, message)
    }

    /// Shorthand for a [`ErrorKind::InvalidRequest`] error.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidRequest, message)
    }

    /// Shorthand for a [`ErrorKind::MalformedResponse`] error.
    pub fn malformed(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::MalformedResponse, message)
    }

    /// Attach a diagnostic origin, replacing any previous one.
    #[must_use = "error modifiers return an updated error"]
    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        self.inner.origin = Some(origin.into());
        self
    }

    #[must_use = "error modifiers return an updated error"]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.inner.model = Some(model.into());
        self
    }

    #[must_use = "error modifiers return an updated error"]
    pub fn with_status(mut self, status: u16) -> Self {
        self.inner.status = Some(status);
        self
    }

    #[must_use = "error modifiers return an updated error"]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.inner.code = Some(code.into());
        self
    }

    #[must_use = "error modifiers return an updated error"]
    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        self.inner.request_id = Some(id.into());
        self
    }

    #[must_use = "error modifiers return an updated error"]
    pub fn with_retry_after(mut self, after: Duration) -> Self {
        self.inner.retry_after = Some(after);
        self
    }

    #[must_use = "error modifiers return an updated error"]
    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.inner.retryable = Some(retryable);
        self
    }

    #[must_use = "error modifiers return an updated error"]
    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.inner.source = Some(Box::new(source));
        self
    }

    pub fn kind(&self) -> ErrorKind {
        self.inner.kind
    }

    /// Credential-free human-readable message.
    pub fn message(&self) -> &str {
        &self.inner.message
    }

    /// Diagnostic API profile or service origin.
    pub fn origin(&self) -> Option<&str> {
        self.inner.origin.as_deref()
    }

    pub fn model(&self) -> Option<&str> {
        self.inner.model.as_deref()
    }

    /// HTTP status code, when the error came from an HTTP response.
    pub fn status(&self) -> Option<u16> {
        self.inner.status
    }

    /// Upstream provider error code (e.g. `context_length_exceeded`).
    pub fn code(&self) -> Option<&str> {
        self.inner.code.as_deref()
    }

    /// Provider request id (from response headers or error body), for support tickets.
    pub fn request_id(&self) -> Option<&str> {
        self.inner.request_id.as_deref()
    }

    /// Provider-suggested wait before retrying, when advertised.
    pub fn retry_after(&self) -> Option<Duration> {
        self.inner.retry_after
    }

    /// Copy every field except the boxed source.
    #[must_use]
    pub fn clone_without_source(&self) -> Self {
        Self {
            inner: Box::new(ErrorInner {
                kind: self.inner.kind,
                message: self.inner.message.clone(),
                origin: self.inner.origin.clone(),
                model: self.inner.model.clone(),
                status: self.inner.status,
                code: self.inner.code.clone(),
                request_id: self.inner.request_id.clone(),
                retry_after: self.inner.retry_after,
                retryable: self.inner.retryable,
                source: None,
            }),
        }
    }

    /// Whether retrying the identical request may succeed.
    ///
    /// Rate limits, overloads, transport failures, timeouts, and 5xx responses
    /// are retryable by default. Errors raised after a stream was established
    /// (truncation, transport failures, idle timeouts) are not, because
    /// partial output may already have been consumed.
    pub fn retryable(&self) -> bool {
        if let Some(explicit) = self.inner.retryable {
            return explicit;
        }
        match self.inner.kind {
            ErrorKind::RateLimited
            | ErrorKind::Overloaded
            | ErrorKind::Transport
            | ErrorKind::Timeout => true,
            ErrorKind::Provider => matches!(self.inner.status, Some(s) if s >= 500),
            _ => false,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = &self.inner;
        write!(f, "{:?}", inner.kind)?;
        if let Some(origin) = &inner.origin {
            write!(f, " [{origin}")?;
            if let Some(model) = &inner.model {
                write!(f, "/{model}")?;
            }
            write!(f, "]")?;
        }
        write!(f, ": {}", inner.message)?;
        if let Some(status) = inner.status {
            write!(f, " (status {status}")?;
            if let Some(code) = &inner.code {
                write!(f, ", code {code}")?;
            }
            if let Some(id) = &inner.request_id {
                write!(f, ", request {id}")?;
            }
            write!(f, ")")?;
        } else if let Some(code) = &inner.code {
            write!(f, " (code {code})")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Error")
            .field("kind", &self.inner.kind)
            .field("message", &self.inner.message)
            .field("origin", &self.inner.origin)
            .field("model", &self.inner.model)
            .field("status", &self.inner.status)
            .field("code", &self.inner.code)
            .field("request_id", &self.inner.request_id)
            .field("retry_after", &self.inner.retry_after)
            .field("retryable", &self.retryable())
            .finish()
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.inner
            .source
            .as_deref()
            .map(|s| s as &(dyn std::error::Error + 'static))
    }
}
