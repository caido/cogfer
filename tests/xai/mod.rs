//! Wire-level tests for the xAI provider presets and OAuth machinery,
//! against the mock transport.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use llmwire::oauth::xai::{XaiAuthenticator, XaiOAuth, XaiTokens};
use llmwire::transport::mock::MockTransport;
use llmwire::transport::{HeaderName, HttpRequest};
use llmwire::{
    ApiProfile, Credentials, DevicePoll, Error, ErrorKind, FinishReason, OAuthStatus,
    ProviderConfig, RequestAuthenticator, TokenStore,
};
use serde_json::json;
use url::Url;

use crate::common::*;

fn responses_completed(model: &str) -> serde_json::Value {
    json!({
        "id": "resp_x1",
        "object": "response",
        "status": "completed",
        "model": model,
        "output": [
            {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "ok", "annotations": []}]}
        ],
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                  "input_tokens_details": {"cached_tokens": 0}}
    })
}

fn chat_completed(model: &str) -> serde_json::Value {
    json!({
        "id": "chatcmpl_x1",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": "ok"}}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
    })
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

fn form_body(request: &llmwire::transport::HttpRequest) -> String {
    String::from_utf8(request.body.clone().expect("form body").to_vec()).unwrap()
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
    async fn save(&self, tokens: &T) -> llmwire::Result<()> {
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
