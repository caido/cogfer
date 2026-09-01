//! OpenAI models on Amazon Bedrock's OpenAI-compatible Responses endpoint:
//! URL and body shape, SSE streaming, both error envelopes, and the
//! model-to-profile mapping.

use std::sync::Arc;

use llmwire::transport::HttpRequest;
use llmwire::transport::mock::MockTransport;
use llmwire::{
    ApiProfile, Credentials, ErrorKind, Message, Provider, ProviderConfig, ReasoningConfig,
    ReasoningEffort, Request,
};
use serde_json::json;

use crate::common::{
    assert_terminal_contract, collect, drain, header, headers, provider_with, text_request,
};

const MODEL: &str = "us.openai.gpt-5.6-sol";

fn bedrock_openai(mock: &Arc<MockTransport>) -> Provider {
    provider_with(
        mock,
        ProviderConfig::bedrock_openai("eu-west-1", Credentials::bearer("bedrock-api-key"))
            .expect("region is valid"),
    )
}

fn completed(text: &str) -> serde_json::Value {
    json!({
        "id": "resp_1", "object": "response", "status": "completed", "model": MODEL,
        "output": [
            {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": text, "annotations": []}]}
        ],
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    })
}

#[tokio::test]
async fn requests_use_the_responses_paths_and_bedrock_auth() {
    let mock = MockTransport::shared();
    mock.push_json(200, &completed("ok"));
    let request = Request::builder()
        .system("Be brief.")
        .message(Message::user("hi"))
        .reasoning(ReasoningConfig::effort(ReasoningEffort::Low))
        .build();

    bedrock_openai(&mock)
        .language_model(MODEL)
        .generate(request)
        .await
        .expect("generate succeeds");

    let http: &HttpRequest = &mock.requests()[0];
    assert_eq!(
        http.url.as_str(),
        "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1/responses"
    );
    assert_eq!(
        header(http, "authorization"),
        Some("Bearer bedrock-api-key")
    );
    let body = mock.request_json(0);
    assert_eq!(body["model"], MODEL);
    assert_eq!(body["store"], false);
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(body["reasoning"]["effort"], "low");
    assert_eq!(body["input"][0]["role"], "system");
    assert!(body.get("stream").is_none(), "{body}");
}

#[tokio::test]
async fn streaming_speaks_sse_without_stream_options() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_1","model":"us.openai.gpt-5.6-sol","status":"in_progress"},"sequence_number":0}"#,
        r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress","content":[]},"sequence_number":1}"#,
        r#"{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hi","sequence_number":2}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hi","annotations":[]}]},"sequence_number":3}"#,
        r#"{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12}},"sequence_number":4}"#,
    ]);

    let events = drain(
        bedrock_openai(&mock)
            .language_model(MODEL)
            .stream(text_request("hi"))
            .await
            .expect("stream establishes"),
    )
    .await;

    assert_terminal_contract(&events);
    let result = collect(&events).expect("stream succeeds");
    assert_eq!(result.text(), "Hi");
    let body = mock.request_json(0);
    assert_eq!(body["stream"], true);
    assert!(body.get("stream_options").is_none(), "{body}");
}

#[tokio::test]
async fn aws_error_envelopes_are_classified_by_exception() {
    let mock = MockTransport::shared();
    mock.push_response(
        429,
        headers(&[
            ("content-type", "application/json"),
            (
                "x-amzn-errortype",
                "ThrottlingException:http://internal.amazon.com/coral/",
            ),
            ("x-amzn-requestid", "req-1"),
        ]),
        r#"{"message":"Too many requests"}"#,
    );

    let error = bedrock_openai(&mock)
        .language_model(MODEL)
        .generate(text_request("hi"))
        .await
        .expect_err("throttled");

    assert_eq!(error.kind(), ErrorKind::RateLimited);
    assert_eq!(error.code(), Some("ThrottlingException"));
    assert_eq!(error.status(), Some(429));
    assert_eq!(error.request_id(), Some("req-1"));
    assert_eq!(error.message(), "Too many requests");
    assert_eq!(error.origin(), Some("bedrock-openai-responses"));
}

#[tokio::test]
async fn openai_error_envelopes_still_decode() {
    let mock = MockTransport::shared();
    mock.push_json(
        400,
        &json!({"error": {"message": "Unknown parameter: foo", "type": "invalid_request_error",
                          "code": "unknown_parameter"}}),
    );

    let error = bedrock_openai(&mock)
        .language_model(MODEL)
        .generate(text_request("hi"))
        .await
        .expect_err("invalid request");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(error.code(), Some("unknown_parameter"));
    assert_eq!(error.message(), "Unknown parameter: foo");
    assert_eq!(error.origin(), Some("bedrock-openai-responses"));
}

#[test]
fn model_ids_map_to_their_bedrock_profile() {
    for model in [
        "openai.gpt-oss-120b",
        "us.openai.gpt-5.6-sol",
        "global.openai.gpt-5.6-terra",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.openai.gpt-5.6-sol",
    ] {
        assert_eq!(
            ApiProfile::for_bedrock_model(model),
            Some(ApiProfile::BedrockOpenAiResponses),
            "{model}"
        );
    }
    for model in [
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-sonnet-4-6",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/eu.anthropic.claude-sonnet-4-6",
    ] {
        assert_eq!(
            ApiProfile::for_bedrock_model(model),
            Some(ApiProfile::BedrockAnthropic),
            "{model}"
        );
    }
    for model in ["meta.llama3-70b-instruct-v1:0", "gpt-5-mini", ""] {
        assert_eq!(ApiProfile::for_bedrock_model(model), None, "{model}");
    }
}

/// Bedrock's endpoint spells the function-call item's `id` as `item_id` on
/// the `output_item.done` event, which must still complete the open call.
#[tokio::test]
async fn function_calls_complete_despite_the_item_id_spelling() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"type":"response.created","response":{"id":"resp_1","model":"m","status":"in_progress"},"sequence_number":0}"#,
        r#"{"type":"response.output_item.added","output_index":0,"item":{"arguments":"","call_id":"call_1","id":"fc_1","name":"get_weather","status":"in_progress","type":"function_call"},"sequence_number":1}"#,
        r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\"location\":\"Paris\"}","sequence_number":2}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"arguments":"{\"location\":\"Paris\"}","call_id":"call_1","item_id":"fc_1","name":"get_weather","output_index":0,"status":"completed","type":"function_call"},"sequence_number":3}"#,
        r#"{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}},"sequence_number":4}"#,
    ]);

    let events = drain(
        bedrock_openai(&mock)
            .language_model(MODEL)
            .stream(crate::common::tool_request("weather?"))
            .await
            .expect("stream establishes"),
    )
    .await;

    assert_terminal_contract(&events);
    let result = collect(&events).expect("stream succeeds");
    assert_eq!(result.finish.reason, llmwire::FinishReason::ToolCalls);
    let call = result.tool_calls().next().expect("one tool call");
    assert_eq!(call.call_id, "call_1");
    assert_eq!(call.item_id.as_deref(), Some("fc_1"));
    assert_eq!(call.arguments, r#"{"location":"Paris"}"#);
}

#[tokio::test]
async fn verify_steps_out_of_the_openai_prefix() {
    let mock = MockTransport::shared();
    mock.push_json(200, &json!({"asyncInvokeSummaries": []}));

    bedrock_openai(&mock)
        .verify()
        .await
        .expect("a valid key verifies");

    let http: &HttpRequest = &mock.requests()[0];
    assert_eq!(
        http.url.as_str(),
        "https://bedrock-runtime.eu-west-1.amazonaws.com/async-invoke?maxResults=1"
    );
    assert_eq!(
        header(http, "authorization"),
        Some("Bearer bedrock-api-key")
    );
}
