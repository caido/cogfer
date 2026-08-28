//! Wire-level conformance tests for the ChatGPT subscription protocol and
//! its OAuth machinery, against the mock transport.

use std::sync::{Arc, Mutex};

use caido_ai::oauth::chatgpt::{ChatGptAuthenticator, ChatGptOAuth, ChatGptTokens};
use caido_ai::transport::mock::MockTransport;
use caido_ai::transport::{HeaderName, HeaderValue};
use caido_ai::{
    Credentials, DevicePoll, Error, ErrorKind, FinishReason, Message, OAuthStatus, ProviderConfig,
    Request, StreamEvent, TokenStore,
};
use serde_json::json;
use url::Url;

use crate::common::*;

fn completed_transcript() -> Vec<&'static str> {
    vec![
        r#"{"type":"response.created","response":{"id":"resp_c1","model":"gpt-5.6-sol","status":"in_progress"}}"#,
        r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress","content":[]}}"#,
        r#"{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"ok"}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok","annotations":[]}]}}"#,
        r#"{"type":"response.completed","response":{"id":"resp_c1","status":"completed","model":"gpt-5.6-sol","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok","annotations":[]}]}],"usage":{"input_tokens":12,"output_tokens":3,"total_tokens":15,"input_tokens_details":{"cached_tokens":0}}}}"#,
        "[DONE]",
    ]
}

fn unpadded_base64url(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let mut buffer = [0u8; 3];
        buffer[..chunk.len()].copy_from_slice(chunk);
        let word = ((buffer[0] as u32) << 16) | ((buffer[1] as u32) << 8) | (buffer[2] as u32);
        for (index, quad) in [word >> 18, word >> 12, word >> 6, word].iter().enumerate() {
            if index <= chunk.len() {
                out.push(ALPHABET[(*quad & 63) as usize] as char);
            }
        }
    }
    out
}

fn jwt(expires_at: i64, account_id: &str) -> String {
    let claims = json!({
        "exp": expires_at,
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    });
    format!(
        "{}.{}.sig",
        unpadded_base64url(br#"{"alg":"RS256"}"#),
        unpadded_base64url(claims.to_string().as_bytes()),
    )
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

struct RecordingTokenStore<T> {
    saved: Mutex<Option<T>>,
    fail: bool,
}

impl<T> Default for RecordingTokenStore<T> {
    fn default() -> Self {
        Self {
            saved: Mutex::new(None),
            fail: false,
        }
    }
}

impl<T: Clone> RecordingTokenStore<T> {
    fn failing() -> Self {
        Self {
            saved: Mutex::new(None),
            fail: true,
        }
    }

    fn saved(&self) -> Option<T> {
        self.saved.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl<T: Clone + Send + Sync> TokenStore<T> for RecordingTokenStore<T> {
    async fn save(&self, tokens: &T) -> caido_ai::Result<()> {
        if self.fail {
            return Err(Error::new(ErrorKind::Provider, "token store failed"));
        }
        *self.saved.lock().unwrap() = Some(tokens.clone());
        Ok(())
    }
}

mod authenticator;
mod oauth;
mod protocol;
