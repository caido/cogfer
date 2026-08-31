//! Provider-neutral scenarios run by every replay suite.
//!
//! Recording runs them against the live provider and replay runs them against
//! the cassette, so the same assertions hold in both. Prompts steer models
//! toward short, checkable answers. Assertions stay loose enough for any
//! provider (non-empty text, finish reason, usage present) and never depend on
//! exact wording.

use llmwire::{
    ErrorKind, FinishReason, Message, Provider, ReasoningConfig, ReasoningEffort, ReasoningOutput,
    Request, StreamEvent, StructuredOutput,
};
use serde_json::json;

use crate::common::{
    agentic_loop, assert_terminal_contract, collect, drain, loop_request, text_request,
};

/// A blocking completion: text, finish reason, usage, and response identity.
pub(crate) async fn generate_text(provider: &Provider, model: &str) {
    let result = provider
        .language_model(model)
        .generate(
            Request::builder()
                .system("Answer with exactly one word.")
                .message(Message::user("Say OK."))
                .build(),
        )
        .await
        .expect("generate succeeds");

    assert!(!result.text().trim().is_empty(), "empty text: {result:?}");
    assert_eq!(result.finish.reason, FinishReason::Stop, "{result:?}");
    assert!(result.usage.total_input_tokens().is_some(), "{result:?}");
    assert!(result.usage.output_tokens.is_some(), "{result:?}");
    assert!(result.response.id.is_some(), "no response id: {result:?}");
}

/// A streamed completion: the terminal contract, text deltas, and the
/// accumulated result. The prompt invites multi-byte characters so recorded
/// chunks exercise UTF-8 handling.
pub(crate) async fn stream_text(provider: &Provider, model: &str) {
    let events = drain(
        provider
            .language_model(model)
            .stream(text_request(
                "Count from 1 to 5 in words, one per line, then add a ✅.",
            ))
            .await
            .expect("stream establishes"),
    )
    .await;

    assert_terminal_contract(&events);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta { .. })),
        "no text deltas"
    );
    let result = collect(&events).expect("stream result");
    assert!(!result.text().trim().is_empty(), "empty text: {result:?}");
    assert_eq!(result.finish.reason, FinishReason::Stop, "{result:?}");
    assert!(result.usage.output_tokens.is_some(), "{result:?}");
}

/// A streamed tool call followed by a blocking turn that consumes the tool
/// result, with reasoning enabled as an agent would.
pub(crate) async fn stream_tool_loop(provider: &Provider, model: &str) {
    let mut request = loop_request();
    request.reasoning = Some(ReasoningConfig::effort(ReasoningEffort::Low));
    agentic_loop(&provider.language_model(model), request).await;
}

/// [`stream_tool_loop`] with reasoning explicitly disabled, for APIs that
/// reject function tools unless a reasoning effort is set (OpenAI `gpt-5.x` on
/// Chat Completions).
pub(crate) async fn stream_tool_loop_reasoning_disabled(provider: &Provider, model: &str) {
    let mut request = loop_request();
    request.reasoning = Some(ReasoningConfig::Disabled);
    agentic_loop(&provider.language_model(model), request).await;
}

/// JSON-schema constrained output parsed back into a value.
pub(crate) async fn structured_output(provider: &Provider, model: &str) {
    let result = provider
        .language_model(model)
        .generate(
            Request::builder()
                .message(Message::user(
                    "Give the city and country of the Eiffel Tower.",
                ))
                .structured_output(StructuredOutput::new(
                    "landmark",
                    json!({
                        "type": "object",
                        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
                        "required": ["city", "country"],
                        "additionalProperties": false
                    }),
                ))
                .build(),
        )
        .await
        .expect("structured output succeeds");

    assert_eq!(result.finish.reason, FinishReason::Stop, "{result:?}");
    let parsed: serde_json::Value = result.structured_output().expect("valid JSON output");
    assert_eq!(
        parsed["city"].as_str().map(str::to_lowercase).as_deref(),
        Some("paris"),
        "{parsed}"
    );
}

/// Visible reasoning in a stream: reasoning parts or reasoning tokens, and a
/// correct answer after them.
pub(crate) async fn reasoning_stream(provider: &Provider, model: &str) {
    let events = drain(
        provider
            .language_model(model)
            .stream(
                Request::builder()
                    .message(Message::user(
                        "Compute 47 * 83 + 19. Think it through step by step, \
                         then answer with just the number.",
                    ))
                    .reasoning(ReasoningConfig::Effort {
                        effort: ReasoningEffort::Medium,
                        output: Some(ReasoningOutput::Include),
                    })
                    .build(),
            )
            .await
            .expect("stream establishes"),
    )
    .await;

    assert_terminal_contract(&events);
    let result = collect(&events).expect("stream result");
    let reasoning_parts = result.reasoning().count();
    let reasoning_tokens = result.usage.reasoning_tokens.unwrap_or(0);
    assert!(
        reasoning_parts > 0 || reasoning_tokens > 0,
        "no evidence of reasoning: {result:?}"
    );
    assert!(result.text().contains("3920"), "wrong answer: {result:?}");
    assert_eq!(result.finish.reason, FinishReason::Stop, "{result:?}");
}

/// The provider's error envelope for a model that does not exist, decoded
/// into the normalized error with origin and model attribution.
pub(crate) async fn unknown_model(provider: &Provider, _model: &str) {
    let error = provider
        .language_model("this-model-does-not-exist")
        .generate(text_request("hi"))
        .await
        .expect_err("unknown model is rejected");

    assert!(
        matches!(
            error.kind(),
            ErrorKind::NotFound | ErrorKind::InvalidRequest
        ),
        "{error:?}"
    );
    assert!(matches!(error.status(), Some(400..=404)), "{error:?}");
    assert_eq!(
        error.origin(),
        Some(provider.profile().as_str()),
        "{error:?}"
    );
    assert_eq!(
        error.model(),
        Some("this-model-does-not-exist"),
        "{error:?}"
    );
}
