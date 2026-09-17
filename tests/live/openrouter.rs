use cogfer::{Message, ProviderConfig, ReasoningConfig, Request, ToolChoice};

use crate::common::{agentic_loop, loop_request, provider_from_env, time_tool};

#[tokio::test]
#[ignore = "live"]
async fn openrouter_generate_with_usage_accounting() {
    let Some(provider) = provider_from_env("OPENROUTER_API_KEY", ProviderConfig::openrouter) else {
        return;
    };
    let request = Request::builder()
        .message(Message::user("Reply with exactly: pong"))
        .max_output_tokens(50)
        .build();
    let result = provider
        .language_model("openai/gpt-5.6-luna")
        .generate(request)
        .await
        .expect("generate succeeds");
    assert!(result.text().to_lowercase().contains("pong"));
    assert!(result.usage.input_tokens.is_some());
    assert!(result.response.id.is_some());
    let metadata = result
        .provider_metadata
        .get("openrouter")
        .expect("OpenRouter response metadata");
    assert!(metadata["provider"].is_string(), "{metadata}");
    assert!(metadata["usage"]["cost"].is_number(), "{metadata}");
    assert!(metadata["usage"]["cost_details"].is_object(), "{metadata}");
    assert!(metadata["usage"]["is_byok"].is_boolean(), "{metadata}");
}

#[tokio::test]
#[ignore = "live"]
async fn openrouter_reasoning_details_round_trip() {
    let Some(provider) = provider_from_env("OPENROUTER_API_KEY", ProviderConfig::openrouter) else {
        return;
    };
    let model = provider.language_model("anthropic/claude-haiku-4.5");
    let mut request = loop_request();
    request.reasoning = Some(ReasoningConfig::budget(
        std::num::NonZeroU32::new(1024).unwrap(),
    ));
    agentic_loop(&model, request).await;
}

#[tokio::test]
#[ignore = "live"]
async fn openrouter_forced_tool_choice() {
    let Some(provider) = provider_from_env("OPENROUTER_API_KEY", ProviderConfig::openrouter) else {
        return;
    };
    let request = Request::builder()
        .message(Message::user("What time is it in Tokyo?"))
        .tool(time_tool())
        .tool_choice(ToolChoice::Tool {
            name: "get_current_time".into(),
        })
        .max_output_tokens(500)
        .build();
    let result = provider
        .language_model("openai/gpt-5.6-luna")
        .generate(request)
        .await
        .expect("generate succeeds");
    assert!(result.has_tool_calls());
}

#[tokio::test]
#[ignore = "live"]
async fn openrouter_streaming_tool_loop() {
    let Some(provider) = provider_from_env("OPENROUTER_API_KEY", ProviderConfig::openrouter) else {
        return;
    };
    let model = provider.language_model("openai/gpt-5.6-luna");
    agentic_loop(&model, loop_request()).await;
}
