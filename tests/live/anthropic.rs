use llmwire::{
    Message, ProviderConfig, ReasoningConfig, ReasoningEffort, Request, StreamEvent,
    StructuredOutput,
};
use serde_json::json;

use crate::common::{
    agentic_loop, assert_terminal_contract, drain, loop_request, provider_from_env, time_tool,
};

#[tokio::test]
#[ignore = "live"]
async fn anthropic_agentic_loop_with_thinking() {
    let Some(provider) = provider_from_env("ANTHROPIC_API_KEY", ProviderConfig::anthropic) else {
        return;
    };
    let model = provider.language_model("claude-haiku-4-5");
    let mut request = loop_request();
    request.reasoning = Some(ReasoningConfig::budget(
        std::num::NonZeroU32::new(1024).unwrap(),
    ));
    agentic_loop(&model, request).await;
}

#[tokio::test]
#[ignore = "live"]
async fn anthropic_streaming_text() {
    let Some(provider) = provider_from_env("ANTHROPIC_API_KEY", ProviderConfig::anthropic) else {
        return;
    };
    let request = Request::builder()
        .message(Message::user("Reply with exactly: pong"))
        .max_output_tokens(50)
        .build();
    let events = drain(
        provider
            .language_model("claude-haiku-4-5")
            .stream(request)
            .await
            .expect("stream establishes"),
    )
    .await;
    assert_terminal_contract(&events);
    let text: String = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::TextDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert!(text.to_lowercase().contains("pong"), "got {text:?}");
}

#[tokio::test]
#[ignore = "live"]
async fn structured_output_with_tools_available() {
    let Some(provider) = provider_from_env("ANTHROPIC_API_KEY", ProviderConfig::anthropic) else {
        return;
    };
    let request = Request::builder()
        .message(Message::user(
            "Return the city and country of the Eiffel Tower as JSON.",
        ))
        .tool(time_tool())
        .structured_output(StructuredOutput::new(
            "landmark",
            json!({
                "type": "object",
                "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
                "required": ["city", "country"],
                "additionalProperties": false
            }),
        ))
        .max_output_tokens(2000)
        .build();
    let result = provider
        .language_model("claude-sonnet-4-6")
        .generate(request)
        .await
        .expect("structured output with tools");
    if !result.has_tool_calls() {
        let parsed: serde_json::Value = result.structured_output().expect("valid JSON");
        assert!(parsed.get("city").is_some(), "got {parsed}");
    }
}

#[tokio::test]
#[ignore = "live"]
async fn anthropic_newest_model_multi_turn_agentic_loop() {
    let Some(provider) = provider_from_env("ANTHROPIC_API_KEY", ProviderConfig::anthropic) else {
        return;
    };
    let model = provider.language_model("claude-opus-5");
    let mut request = loop_request();
    request.reasoning = Some(ReasoningConfig::effort(ReasoningEffort::Low));
    agentic_loop(&model, request).await;
}

#[tokio::test]
#[ignore = "live"]
async fn anthropic_effort_actually_enables_thinking() {
    let Some(provider) = provider_from_env("ANTHROPIC_API_KEY", ProviderConfig::anthropic) else {
        return;
    };
    for model_id in ["claude-opus-4-8", "claude-opus-5"] {
        let request = Request::builder()
            .message(Message::user("What is 47*93? Think it through."))
            .reasoning(ReasoningConfig::effort(ReasoningEffort::High))
            .max_output_tokens(3000)
            .build();
        let result = provider
            .language_model(model_id)
            .generate(request)
            .await
            .unwrap_or_else(|error| panic!("{model_id}: {error}"));
        let reasoning_tokens = result.usage.reasoning_tokens.unwrap_or(0);
        assert!(
            reasoning_tokens > 0 || result.reasoning().next().is_some(),
            "{model_id}: reasoning was requested but the model did not think \
             (reasoning_tokens={reasoning_tokens})"
        );
        eprintln!("{model_id}: reasoning_tokens={reasoning_tokens}");
    }
}

#[tokio::test]
#[ignore = "live"]
async fn anthropic_xhigh_effort_is_accepted() {
    let Some(provider) = provider_from_env("ANTHROPIC_API_KEY", ProviderConfig::anthropic) else {
        return;
    };
    let request = Request::builder()
        .message(Message::user("Reply with the single word: ok"))
        .reasoning(ReasoningConfig::effort(ReasoningEffort::XHigh))
        .max_output_tokens(2048)
        .build();
    let result = provider
        .language_model("claude-sonnet-5")
        .generate(request)
        .await
        .expect("xhigh must pass through unchanged");
    assert!(
        result
            .warnings
            .iter()
            .all(|warning| warning.subject.as_deref() != Some("reasoning.effort")),
        "{:?}",
        result.warnings
    );
}
