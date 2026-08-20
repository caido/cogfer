use super::ApiProfile;
use crate::error::{Error, ErrorKind};

/// Build the common fallback for a non-JSON or unrecognized provider error.
pub(crate) fn fallback_provider_error(protocol: ApiProfile, status: u16, body: &[u8]) -> Error {
    Error::new(
        crate::http::error_kind_for_status(status),
        fallback_provider_error_message(protocol, status, body),
    )
    .with_origin(protocol.as_str())
}

pub(crate) fn fallback_provider_error_message(
    protocol: ApiProfile,
    status: u16,
    body: &[u8],
) -> String {
    format!(
        "{} returned HTTP {status}: {}",
        protocol.as_str(),
        crate::util::truncate_for_error(&String::from_utf8_lossy(body), 200)
    )
}

pub(crate) fn is_content_policy_code(code: &str) -> bool {
    matches!(
        code.to_ascii_lowercase().as_str(),
        "content_policy_violation"
            | "image_content_policy_violation"
            | "content_filter"
            | "refusal"
            | "safety"
            | "safety_refusal"
            | "cyber_abuse"
            | "cyber_safety"
            | "blocklist"
            | "prohibited_content"
            | "image_safety"
            | "image_prohibited_content"
    )
}

pub(crate) fn content_policy_kind(code: Option<&str>, fallback: ErrorKind) -> ErrorKind {
    if code.is_some_and(is_content_policy_code) {
        ErrorKind::ContentPolicy
    } else {
        fallback
    }
}
