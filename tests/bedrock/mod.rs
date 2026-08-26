//! Anthropic on Amazon Bedrock: URL and body shape, buffered and event-stream
//! responses, stream exceptions, and AWS error envelopes.

use std::sync::Arc;

use caido_ai::transport::HttpRequest;
use caido_ai::transport::mock::MockTransport;
use caido_ai::{
    Credentials, ErrorKind, FinishReason, Message, Provider, ProviderConfig, ReasoningConfig,
    ReasoningEffort, Request, StreamEvent, WarningKind,
};
use serde_json::json;

use crate::common::{
    assert_terminal_contract, collect, drain, header, headers, provider_with, text_request,
};

const MODEL: &str = "anthropic.claude-sonnet-4-5-20250929-v1:0";

fn bedrock(mock: &Arc<MockTransport>) -> Provider {
    provider_with(
        mock,
        ProviderConfig::bedrock_anthropic("eu-west-1", Credentials::bearer("bedrock-api-key")),
    )
}

fn message(text: &str) -> serde_json::Value {
    json!({
        "id": "msg_b1", "type": "message", "role": "assistant", "model": MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 3}
    })
}

/// Standard base64, as Bedrock wraps each model event.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let mut buffer = [0u8; 3];
        buffer[..chunk.len()].copy_from_slice(chunk);
        let value = u32::from_be_bytes([0, buffer[0], buffer[1], buffer[2]]);
        for index in 0..4 {
            if index <= chunk.len() {
                out.push(ALPHABET[((value >> (18 - 6 * index)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// One Bedrock `chunk` event carrying an Anthropic stream event.
fn chunk(event: &str) -> Vec<u8> {
    json!({"bytes": base64(event.as_bytes()), "p": "abc"})
        .to_string()
        .into_bytes()
}

fn events() -> Vec<Vec<u8>> {
    [
        r#"{"type":"message_start","message":{"id":"msg_b1","type":"message","role":"assistant","model":"claude","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":1}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}"#,
        r#"{"type":"message_stop","amazon-bedrock-invocationMetrics":{"inputTokenCount":10,"outputTokenCount":3,"invocationLatency":50,"firstByteLatency":20}}"#,
    ]
    .iter()
    .map(|event| chunk(event))
    .collect()
}

fn as_chunks(events: &[Vec<u8>]) -> Vec<(&str, &[u8])> {
    events
        .iter()
        .map(|payload| ("chunk", payload.as_slice()))
        .collect()
}

#[tokio::test]
async fn requests_use_bedrock_urls_and_body_conventions() {
    let mock = MockTransport::shared();
    mock.push_json(200, &message("ok"));
    let request = Request::builder()
        .system("Be brief.")
        .message(Message::user("hi"))
        .reasoning(ReasoningConfig::effort(ReasoningEffort::Low))
        .build();

    bedrock(&mock)
        .language_model(MODEL)
        .generate(request)
        .await
        .expect("generate succeeds");

    let http: &HttpRequest = &mock.requests()[0];
    assert_eq!(
        http.url.as_str(),
        "https://bedrock-runtime.eu-west-1.amazonaws.com/model/anthropic.claude-sonnet-4-5-20250929-v1%3A0/invoke"
    );
    assert_eq!(
        header(http, "authorization"),
        Some("Bearer bedrock-api-key")
    );
    assert_eq!(header(http, "accept"), Some("application/json"));
    assert_eq!(header(http, "content-type"), Some("application/json"));
    assert!(header(http, "anthropic-version").is_none());
    let body = mock.request_json(0);
    assert_eq!(body["anthropic_version"], "bedrock-2023-05-31");
    assert!(body.get("model").is_none(), "{body}");
    assert!(body.get("stream").is_none(), "{body}");
    assert_eq!(body["system"], "Be brief.");
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert_eq!(body["output_config"]["effort"], "low");
}

#[tokio::test]
async fn streaming_uses_the_event_stream_endpoint_and_decodes_chunks() {
    let mock = MockTransport::shared();
    let events = events();
    mock.push_event_stream(&as_chunks(&events));

    let stream = bedrock(&mock)
        .language_model(MODEL)
        .stream(text_request("hi"))
        .await
        .expect("stream establishes");
    let events = drain(stream).await;

    assert!(
        mock.requests()[0]
            .url
            .path()
            .ends_with("/invoke-with-response-stream")
    );
    assert_terminal_contract(&events);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta { delta, .. } if delta == "Hi"))
    );
    let result = collect(&events).expect("stream result");
    assert_eq!(result.text(), "Hi");
    assert_eq!(result.finish.reason, FinishReason::Stop);
    assert_eq!(result.usage.output_tokens, Some(3));
    assert_eq!(result.response.id.as_deref(), Some("msg_b1"));
}

#[tokio::test]
async fn stream_exceptions_end_the_stream_with_a_classified_error() {
    let mock = MockTransport::shared();
    let events = events();
    mock.push_event_stream_then_exception(
        &as_chunks(&events[..3]),
        "throttlingException",
        br#"{"message":"Too many requests"}"#,
    );

    let events = drain(
        bedrock(&mock)
            .language_model(MODEL)
            .stream(text_request("hi"))
            .await
            .expect("stream establishes"),
    )
    .await;

    assert_terminal_contract(&events);
    let error = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::Error { error } => Some(error),
            _ => None,
        })
        .expect("error event");
    assert_eq!(error.kind(), ErrorKind::RateLimited);
    assert_eq!(error.message(), "Too many requests");
    assert_eq!(error.code(), Some("throttlingException"));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta { delta, .. } if delta == "Hi")),
        "output before the exception is delivered"
    );
}

#[tokio::test]
async fn aws_error_envelopes_are_classified() {
    let mock = MockTransport::shared();
    mock.push_response(
        403,
        headers(&[
            ("content-type", "application/json"),
            (
                "x-amzn-errortype",
                "ExpiredTokenException:http://internal.amazon.com/coral/",
            ),
            ("x-amzn-requestid", "req-1"),
        ]),
        r#"{"message":"The security token included in the request is expired"}"#,
    );

    let error = bedrock(&mock)
        .language_model(MODEL)
        .generate(text_request("hi"))
        .await
        .expect_err("expired token");

    assert_eq!(error.kind(), ErrorKind::Authentication);
    assert_eq!(error.code(), Some("ExpiredTokenException"));
    assert_eq!(error.status(), Some(403));
    assert!(error.message().contains("expired"));
    assert_eq!(error.origin(), Some("bedrock-anthropic"));
}

#[tokio::test]
async fn native_compaction_is_not_available() {
    let mock = MockTransport::shared();
    let request = Request::builder()
        .message(Message::user("hi"))
        .compaction(caido_ai::Compaction::enabled())
        .build();

    let error = bedrock(&mock)
        .language_model(MODEL)
        .generate(request)
        .await
        .expect_err("compaction is an anthropic.com beta");

    assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
    assert!(!bedrock(&mock).capabilities().native_compaction);
    assert!(mock.requests().is_empty());
}

#[tokio::test]
async fn buffered_responses_decode_like_anthropic() {
    let mock = MockTransport::shared();
    mock.push_json(200, &message("OK"));

    let result = bedrock(&mock)
        .language_model(MODEL)
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");

    assert_eq!(result.text(), "OK");
    assert_eq!(result.finish.reason, FinishReason::Stop);
    assert!(
        !result
            .warnings
            .iter()
            .any(|warning| warning.kind == WarningKind::UnsupportedSetting),
        "{:?}",
        result.warnings
    );
}
