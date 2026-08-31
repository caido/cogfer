//! Anthropic on Amazon Bedrock: URL and body shape, buffered and event-stream
//! responses, stream exceptions, and AWS error envelopes.

use std::sync::Arc;

use llmwire::transport::HttpRequest;
use llmwire::transport::mock::MockTransport;
use llmwire::{
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
        ProviderConfig::bedrock_anthropic("eu-west-1", Credentials::bearer("bedrock-api-key"))
            .expect("region is valid"),
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

/// Standard base64, the encoding Bedrock wraps each model event in.
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

/// An `error` message carries its cause as plain header text rather than a
/// JSON envelope. Parsing it as JSON discards the cause and leaves the caller
/// with nothing but the exception class.
#[tokio::test]
async fn stream_errors_report_the_message_from_the_header() {
    let mock = MockTransport::shared();
    let events = events();
    mock.push_event_stream_then_error(
        &as_chunks(&events[..3]),
        "modelStreamErrorException",
        "the model stopped responding",
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
    assert_eq!(error.message(), "the model stopped responding");
    assert_eq!(error.code(), Some("modelStreamErrorException"));
    assert_eq!(error.kind(), ErrorKind::Provider);
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
    assert_eq!(error.request_id(), Some("req-1"));
    assert!(error.message().contains("expired"));
    assert_eq!(error.origin(), Some("bedrock-anthropic"));
}

#[tokio::test]
async fn native_compaction_is_not_available() {
    let mock = MockTransport::shared();
    let request = Request::builder()
        .message(Message::user("hi"))
        .compaction(llmwire::Compaction::enabled())
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
async fn replayed_compaction_history_is_rejected() {
    let mock = MockTransport::shared();
    let request = Request::builder()
        .message(Message::user("hi"))
        .message(Message::Assistant {
            content: vec![llmwire::AssistantPart::Compaction(
                llmwire::CompactionPart {
                    id: None,
                    content: Some("summary".into()),
                    encrypted_content: None,
                },
            )],
            provider_metadata: Default::default(),
        })
        .message(Message::user("continue"))
        .build();

    let error = bedrock(&mock)
        .language_model(MODEL)
        .generate(request)
        .await
        .expect_err("bedrock has no compaction beta");

    assert_eq!(error.kind(), ErrorKind::UnsupportedContent);
    assert!(mock.requests().is_empty());
}

#[test]
fn regions_are_validated() {
    assert!(ProviderConfig::bedrock_anthropic("us east 1", Credentials::none()).is_err());
    assert!(ProviderConfig::bedrock_anthropic("", Credentials::none()).is_err());
    assert!(ProviderConfig::bedrock_anthropic("us-gov-west-1", Credentials::none()).is_ok());
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

#[cfg(feature = "aws")]
mod sigv4 {
    use std::sync::Arc;

    use llmwire::aws::{AwsCredentials, SigV4Authenticator};
    use llmwire::transport::mock::MockTransport;
    use llmwire::{Credentials, ErrorKind, ProviderConfig};

    use super::{MODEL, message};
    use crate::common::{header, headers, provider_with, text_request};

    fn signed_provider(mock: &Arc<MockTransport>) -> llmwire::Provider {
        let credentials =
            AwsCredentials::new("AKIDEXAMPLE", "secret").with_session_token("session");
        provider_with(
            mock,
            ProviderConfig::bedrock_anthropic("eu-west-1", Credentials::none())
                .expect("region is valid")
                .with_authenticator(Arc::new(SigV4Authenticator::new("eu-west-1", credentials))),
        )
    }

    #[tokio::test]
    async fn requests_are_signed_for_the_bedrock_service_in_the_region() {
        let mock = MockTransport::shared();
        mock.push_json(200, &message("ok"));

        signed_provider(&mock)
            .language_model(MODEL)
            .generate(text_request("hi"))
            .await
            .expect("generate succeeds");

        let http = &mock.requests()[0];
        let authorization = header(http, "authorization").expect("signed");
        assert!(
            authorization.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"),
            "{authorization}"
        );
        assert!(
            authorization.contains(
                "/eu-west-1/bedrock/aws4_request, \
                 SignedHeaders=accept;content-type;host;x-amz-date;x-amz-security-token, \
                 Signature="
            ),
            "{authorization}"
        );
        assert_eq!(
            header(http, "host"),
            Some("bedrock-runtime.eu-west-1.amazonaws.com")
        );
        assert!(header(http, "x-amz-date").is_some_and(|date| date.ends_with('Z')));
        assert_eq!(header(http, "x-amz-security-token"), Some("session"));
    }

    #[tokio::test]
    async fn a_rejected_signature_is_recomputed_once() {
        let mock = MockTransport::shared();
        mock.push_response(
            403,
            headers(&[("x-amzn-errortype", "InvalidSignatureException")]),
            r#"{"message":"The request signature we calculated does not match"}"#,
        );
        mock.push_json(200, &message("ok"));

        let result = signed_provider(&mock)
            .language_model(MODEL)
            .generate(text_request("hi"))
            .await
            .expect("the re-signed request succeeds");

        assert_eq!(result.text(), "ok");
        let requests = mock.requests();
        assert_eq!(requests.len(), 2);
        assert!(header(&requests[1], "authorization").is_some());
    }

    /// Credentials that rotate on every fetch, like an STS session would.
    struct Rotating(std::sync::atomic::AtomicU32);

    #[async_trait::async_trait]
    impl llmwire::aws::AwsCredentialsProvider for Rotating {
        async fn credentials(&self) -> llmwire::Result<AwsCredentials> {
            let generation = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            Ok(AwsCredentials::new(format!("AKID{generation}"), "secret"))
        }
    }

    #[tokio::test]
    async fn expired_credentials_are_refetched_before_re_signing() {
        let mock = MockTransport::shared();
        mock.push_response(
            403,
            headers(&[("x-amzn-errortype", "ExpiredTokenException")]),
            r#"{"message":"The security token included in the request is expired"}"#,
        );
        mock.push_json(200, &message("ok"));
        let provider = provider_with(
            &mock,
            ProviderConfig::bedrock_anthropic("eu-west-1", Credentials::none())
                .expect("region is valid")
                .with_authenticator(Arc::new(SigV4Authenticator::with_provider(
                    "eu-west-1",
                    Arc::new(Rotating(Default::default())),
                ))),
        );

        provider
            .language_model(MODEL)
            .generate(text_request("hi"))
            .await
            .expect("the re-signed request succeeds");

        let requests = mock.requests();
        assert!(
            header(&requests[0], "authorization")
                .unwrap()
                .contains("Credential=AKID1/")
        );
        assert!(
            header(&requests[1], "authorization")
                .unwrap()
                .contains("Credential=AKID2/")
        );
    }

    #[tokio::test]
    async fn a_permission_denial_is_not_retried() {
        let mock = MockTransport::shared();
        mock.push_response(
            403,
            headers(&[("x-amzn-errortype", "AccessDeniedException")]),
            r#"{"message":"not allowed"}"#,
        );

        let error = signed_provider(&mock)
            .language_model(MODEL)
            .generate(text_request("hi"))
            .await
            .expect_err("denied");

        assert_eq!(error.kind(), ErrorKind::Permission);
        assert_eq!(mock.requests().len(), 1, "re-signing cannot help");
    }

    #[tokio::test]
    async fn a_second_signature_rejection_is_terminal() {
        let mock = MockTransport::shared();
        for _ in 0..2 {
            mock.push_response(
                403,
                headers(&[("x-amzn-errortype", "RequestTimeTooSkewed")]),
                r#"{"message":"clock skew"}"#,
            );
        }

        let error = signed_provider(&mock)
            .language_model(MODEL)
            .generate(text_request("hi"))
            .await
            .expect_err("rejected twice");

        assert_eq!(error.kind(), ErrorKind::Authentication);
        assert_eq!(mock.requests().len(), 2);
    }
}
