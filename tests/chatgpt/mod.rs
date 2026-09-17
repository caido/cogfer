//! Wire-level conformance tests for the ChatGPT subscription protocol and
//! its OAuth machinery, against the mock transport.

use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use cogfer::oauth::chatgpt::{ChatGptAuthenticator, ChatGptOAuth, ChatGptTokens};
use cogfer::transport::mock::MockTransport;
use cogfer::transport::{HeaderName, HeaderValue};
use cogfer::{
    Credentials, DevicePoll, ErrorKind, FinishReason, Message, OAuthStatus, ProviderConfig,
    Request, StreamEvent,
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

fn jwt(expires_at: i64, account_id: &str) -> String {
    let claims = json!({
        "exp": expires_at,
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    });
    format!(
        "{}.{}.sig",
        URL_SAFE_NO_PAD.encode(br#"{"alg":"RS256"}"#),
        URL_SAFE_NO_PAD.encode(claims.to_string().as_bytes()),
    )
}

mod authenticator;
mod oauth;
mod protocol;
