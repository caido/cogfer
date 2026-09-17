use cogfer::{FinishReason, Message, ProviderConfig, Request};

use crate::common::{
    agentic_loop, assert_terminal_contract, chatgpt_provider_on, drain, live_client, loop_request,
    xai_provider_on,
};

#[tokio::test]
#[ignore = "live"]
async fn chatgpt_generate_and_stream() {
    let Some(provider) = chatgpt_provider_on(&live_client()) else {
        return;
    };
    let model = provider.language_model("gpt-5.6-sol");

    let result = model
        .generate(
            Request::builder()
                .system("Answer with exactly one word.")
                .message(Message::user("Say OK."))
                .build(),
        )
        .await
        .expect("chatgpt generate succeeds");
    println!("chatgpt generate: {:?}", result.text());
    assert!(!result.text().is_empty());
    assert_eq!(result.finish.reason, FinishReason::Stop);
    assert!(result.usage.total_input_tokens().is_some());

    let events = drain(
        model
            .stream(
                Request::builder()
                    .message(Message::user("Count to three, digits only."))
                    .build(),
            )
            .await
            .expect("chatgpt stream establishes"),
    )
    .await;
    assert_terminal_contract(&events);
}

#[tokio::test]
#[ignore = "live"]
async fn chatgpt_agentic_tool_loop() {
    let Some(provider) = chatgpt_provider_on(&live_client()) else {
        return;
    };
    let model = provider.language_model("gpt-5.6-sol");
    agentic_loop(&model, loop_request()).await;
}

#[tokio::test]
#[ignore = "live"]
async fn xai_generate_and_stream() {
    let Some(provider) = xai_provider_on(&live_client(), ProviderConfig::xai) else {
        return;
    };
    let model = provider.language_model("grok-4.5");

    let result = model
        .generate(
            Request::builder()
                .system("Answer with exactly one word.")
                .message(Message::user("Say OK."))
                .build(),
        )
        .await
        .expect("xai generate succeeds");
    println!("xai generate: {:?}", result.text());
    assert!(!result.text().is_empty());
    assert_eq!(result.finish.reason, FinishReason::Stop);
    assert!(result.usage.total_input_tokens().is_some());

    let events = drain(
        model
            .stream(
                Request::builder()
                    .message(Message::user("Count to three, digits only."))
                    .build(),
            )
            .await
            .expect("xai stream establishes"),
    )
    .await;
    assert_terminal_contract(&events);
}

#[tokio::test]
#[ignore = "live"]
async fn xai_chat_dialect_generate() {
    let Some(provider) = xai_provider_on(&live_client(), ProviderConfig::xai_chat) else {
        return;
    };
    let result = provider
        .language_model("grok-4.3")
        .generate(
            Request::builder()
                .system("Answer with exactly one word.")
                .message(Message::user("Say OK."))
                .build(),
        )
        .await
        .expect("xai chat generate succeeds");
    println!("xai chat generate: {:?}", result.text());
    assert!(!result.text().is_empty());
}

#[tokio::test]
#[ignore = "live"]
async fn xai_agentic_tool_loop() {
    let Some(provider) = xai_provider_on(&live_client(), ProviderConfig::xai) else {
        return;
    };
    let model = provider.language_model("grok-4.5");
    agentic_loop(&model, loop_request()).await;
}
