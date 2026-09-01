use std::fmt;

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A ChatGPT OAuth token set.
///
/// Serialization exposes secrets for persistence. `Debug` output is redacted.
#[derive(Clone, Serialize, Deserialize)]
#[must_use = "token modifiers return an updated value"]
pub struct ChatGptTokens {
    pub access_token: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_token: Option<String>,
    /// Unix seconds when the access token expires (the JWT `exp` claim).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    /// The account the subscription belongs to, sent as the
    /// `chatgpt-account-id` header.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
}

impl ChatGptTokens {
    pub fn new(access_token: impl Into<String>) -> Self {
        Self {
            access_token: access_token.into(),
            refresh_token: None,
            id_token: None,
            expires_at: None,
            account_id: None,
        }
    }

    pub fn with_refresh_token(mut self, refresh_token: impl Into<String>) -> Self {
        self.refresh_token = Some(refresh_token.into());
        self
    }

    pub fn with_account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// Fill missing expiry and account id from JWT claims.
    pub fn derive_metadata(mut self) -> Self {
        if self.expires_at.is_none() {
            self.expires_at = jwt_claims(&self.access_token)
                .and_then(|claims| claims.get("exp").and_then(Value::as_i64));
        }
        if self.account_id.is_none() {
            self.account_id = self
                .id_token
                .as_deref()
                .and_then(account_id_claim)
                .or_else(|| account_id_claim(&self.access_token));
        }
        self
    }
}

impl fmt::Debug for ChatGptTokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatGptTokens")
            .field("access_token", &"<redacted>")
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| "<redacted>"),
            )
            .field("id_token", &self.id_token.as_ref().map(|_| "<redacted>"))
            .field("expires_at", &self.expires_at)
            .field("account_id", &self.account_id)
            .finish()
    }
}

/// Decode metadata from a TLS-issued JWT without verifying its signature.
fn jwt_claims(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn account_id_claim(token: &str) -> Option<String> {
    jwt_claims(token)?
        .get("https://api.openai.com/auth")?
        .get("chatgpt_account_id")?
        .as_str()
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jwt_with_claims(claims: &Value) -> String {
        format!(
            "{}.{}.signature",
            URL_SAFE_NO_PAD.encode(br#"{"alg":"RS256"}"#),
            URL_SAFE_NO_PAD.encode(claims.to_string().as_bytes()),
        )
    }

    #[test]
    fn derive_metadata_extracts_expiry_and_account_id() {
        let access = jwt_with_claims(&serde_json::json!({
            "exp": 1_900_000_000_i64,
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_access"},
        }));
        let id = jwt_with_claims(&serde_json::json!({
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_id_token"},
        }));

        let tokens = ChatGptTokens {
            id_token: Some(id),
            ..ChatGptTokens::new(access.clone())
        }
        .derive_metadata();
        assert_eq!(tokens.expires_at, Some(1_900_000_000));
        assert_eq!(tokens.account_id.as_deref(), Some("acct_id_token"));

        let tokens = ChatGptTokens::new(access).derive_metadata();
        assert_eq!(tokens.account_id.as_deref(), Some("acct_access"));
    }

    #[test]
    fn derive_metadata_keeps_existing_values() {
        let tokens = ChatGptTokens::new("opaque-token")
            .with_account_id("acct_explicit")
            .derive_metadata();
        assert_eq!(tokens.account_id.as_deref(), Some("acct_explicit"));
        assert_eq!(tokens.expires_at, None);
    }

    #[test]
    fn debug_output_redacts_tokens() {
        let tokens = ChatGptTokens::new("secret-access")
            .with_refresh_token("secret-refresh")
            .with_account_id("acct_1");
        let debug = format!("{tokens:?}");
        assert!(!debug.contains("secret-access"));
        assert!(!debug.contains("secret-refresh"));
        assert!(debug.contains("acct_1"));
    }

    #[test]
    fn tokens_serialize_real_values_for_persistence() {
        let tokens = ChatGptTokens::new("secret-access").with_refresh_token("secret-refresh");
        let json = serde_json::to_value(&tokens).unwrap();
        assert_eq!(json["access_token"], "secret-access");
        assert_eq!(json["refresh_token"], "secret-refresh");
        let restored: ChatGptTokens = serde_json::from_value(json).unwrap();
        assert_eq!(restored.access_token, "secret-access");
    }
}
