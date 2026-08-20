use caido_ai::{
    Message, ProviderConfig, ReasoningConfig, ReasoningEffort, Request, StructuredOutput,
    ToolResultPart,
};
use serde_json::json;

use crate::common::{agentic_loop, loop_request, provider_from_env, time_tool};

#[tokio::test]
#[ignore = "live"]
async fn openai_responses_agentic_loop() {
    let Some(provider) = provider_from_env("OPENAI_API_KEY", ProviderConfig::openai_responses)
    else {
        return;
    };
    let model = provider.language_model("gpt-5.6-luna");
    let mut request = loop_request();
    request.reasoning = Some(ReasoningConfig::effort(ReasoningEffort::Low));
    agentic_loop(&model, request).await;
}

#[tokio::test]
#[ignore = "live"]
async fn openai_chat_agentic_loop() {
    let Some(provider) = provider_from_env("OPENAI_API_KEY", ProviderConfig::openai_chat) else {
        return;
    };
    let model = provider.language_model("gpt-5.6-luna");
    agentic_loop(&model, loop_request()).await;
}

#[tokio::test]
#[ignore = "live"]
async fn openai_chat_structured_output() {
    let Some(provider) = provider_from_env("OPENAI_API_KEY", ProviderConfig::openai_chat) else {
        return;
    };
    let request = Request::builder()
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
        .max_output_tokens(2000)
        .build();
    let result = provider
        .language_model("gpt-5.6-luna")
        .generate(request)
        .await
        .expect("structured output succeeds");
    let parsed: serde_json::Value = result.structured_output().expect("valid JSON output");
    assert_eq!(
        parsed["city"].as_str().map(str::to_lowercase),
        Some("paris".into())
    );
}

#[tokio::test]
#[ignore = "live"]
async fn compaction_rejected_on_chat_protocol() {
    let Some(provider) = provider_from_env("OPENAI_API_KEY", ProviderConfig::openai_chat) else {
        return;
    };
    let request = Request::builder()
        .message(Message::user("hi"))
        .compaction(caido_ai::Compaction::enabled())
        .build();
    let error = provider
        .language_model("gpt-5.6-luna")
        .generate(request)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), caido_ai::ErrorKind::UnsupportedCapability);
}

#[tokio::test]
#[ignore = "live"]
async fn streaming_and_blocking_agree() {
    let Some(provider) = provider_from_env("OPENAI_API_KEY", ProviderConfig::openai_chat) else {
        return;
    };
    let model = provider.language_model("gpt-5.6-luna");
    let request = Request::builder()
        .message(Message::user(
            "Reply with exactly the three words: alpha beta gamma",
        ))
        .max_output_tokens(2000)
        .build();

    let blocking = model.generate(request.clone()).await.expect("blocking");
    let streamed = model
        .stream(request)
        .await
        .expect("stream")
        .collect_result()
        .await
        .expect("collect");

    assert_eq!(blocking.finish.reason, streamed.finish.reason);
    for text in [
        blocking.text().to_lowercase(),
        streamed.text().to_lowercase(),
    ] {
        assert!(text.contains("alpha"), "unexpected text: {text:?}");
    }
    assert!(streamed.usage.output_tokens.is_some(), "streamed usage");
    assert!(blocking.usage.output_tokens.is_some(), "blocking usage");
}

#[tokio::test]
#[ignore = "live"]
async fn parallel_tool_calls_round_trip() {
    let Some(provider) = provider_from_env("OPENAI_API_KEY", ProviderConfig::openai_responses)
    else {
        return;
    };
    let model = provider.language_model("gpt-5.6-luna");
    let base = Request::builder()
        .system("Call get_current_time once per city, in a single turn.")
        .message(Message::user("What time is it in Paris and in Tokyo?"))
        .tool(time_tool())
        .parallel_tool_calls(true)
        .max_output_tokens(3000)
        .build();

    let first = model.generate(base.clone()).await.expect("turn 1");
    let calls: Vec<_> = first.tool_calls().cloned().collect();
    assert!(!calls.is_empty(), "expected tool calls");
    eprintln!("parallel: {} tool call(s)", calls.len());

    let mut messages = base.messages.clone();
    messages.push(first.to_assistant_message());
    for call in &calls {
        messages.push(Message::tool_result(ToolResultPart::for_call(
            call,
            "It is 12:00 noon.",
        )));
    }
    let mut request = base.clone();
    request.messages = messages;
    let second = model.generate(request).await.expect("turn 2");
    assert!(!second.text().is_empty());
}

#[tokio::test]
#[ignore = "live"]
async fn openai_sub_minimum_compaction_threshold_is_clamped() {
    let Some(provider) = provider_from_env("OPENAI_API_KEY", ProviderConfig::openai_responses)
    else {
        return;
    };
    let request = Request::builder()
        .message(Message::user("Reply with the single word: ok"))
        .compaction(caido_ai::Compaction {
            trigger_input_tokens: Some(500),
            ..caido_ai::Compaction::default()
        })
        .max_output_tokens(64)
        .build();
    let result = provider
        .language_model("gpt-5.6-luna")
        .generate(request)
        .await
        .expect("threshold must be clamped, not sent as-is");
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.message.contains("1000")),
        "{:?}",
        result.warnings
    );
}
