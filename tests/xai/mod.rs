//! Wire-level tests for the SpaceXAI provider presets and OAuth machinery,
//! against the mock transport.

use std::sync::Arc;
use std::time::Duration;

use cogfer::oauth::xai::{XaiAuthenticator, XaiOAuth, XaiTokens};
use cogfer::transport::mock::MockTransport;
use cogfer::transport::{HeaderName, HttpRequest};
use cogfer::{
    ApiProfile, Credentials, DevicePoll, ErrorKind, FinishReason, OAuthStatus, ProviderConfig,
    RequestAuthenticator,
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

fn form_body(request: &cogfer::transport::HttpRequest) -> String {
    String::from_utf8(request.body.clone().expect("form body").to_vec()).unwrap()
}

mod authenticator;
mod oauth;
mod protocol;
