use std::fmt;

use serde::{Deserialize, Serialize};

/// A SpaceXAI OAuth token set.
///
/// Serialization exposes secrets for persistence. `Debug` output is redacted.
#[derive(Clone, Serialize, Deserialize)]
#[must_use = "token modifiers return an updated value"]
pub struct XaiTokens {
    pub access_token: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    /// Unix seconds when the access token expires.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
}

impl XaiTokens {
    pub fn new(access_token: impl Into<String>) -> Self {
        Self {
            access_token: access_token.into(),
            refresh_token: None,
            expires_at: None,
        }
    }

    pub fn with_refresh_token(mut self, refresh_token: impl Into<String>) -> Self {
        self.refresh_token = Some(refresh_token.into());
        self
    }
}

impl fmt::Debug for XaiTokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XaiTokens")
            .field("access_token", &"<redacted>")
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| "<redacted>"),
            )
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_output_redacts_tokens() {
        let tokens = XaiTokens::new("secret-access").with_refresh_token("secret-refresh");
        let debug = format!("{tokens:?}");
        assert!(!debug.contains("secret-access"));
        assert!(!debug.contains("secret-refresh"));
    }

    #[test]
    fn tokens_serialize_real_values_for_persistence() {
        let tokens = XaiTokens::new("secret-access").with_refresh_token("secret-refresh");
        let json = serde_json::to_value(&tokens).unwrap();
        assert_eq!(json["access_token"], "secret-access");
        assert_eq!(json["refresh_token"], "secret-refresh");
        let restored: XaiTokens = serde_json::from_value(json).unwrap();
        assert_eq!(restored.access_token, "secret-access");
    }
}
