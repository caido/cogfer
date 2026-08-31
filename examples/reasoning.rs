//! Reasoning across turns: ask for visible reasoning, print what the provider
//! exposes, then replay the assistant turn (reasoning included) so the next
//! request keeps the model's context. On OpenAI Responses the replayed part
//! carries the encrypted reasoning, and on Anthropic and Gemini it carries
//! the provider signature. `to_assistant_message` handles all of them.
//!
//! ```sh
//! OPENAI_API_KEY=... cargo run --example reasoning
//! ```

use llmwire::{
    Client, Credentials, Message, ProviderConfig, ReasoningConfig, ReasoningEffort,
    ReasoningOutput, Request,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    llmwire::transport::install_default_crypto_provider();
    let client = Client::builder().build()?;
    let provider = client.provider(ProviderConfig::openai_responses(Credentials::api_key(
        std::env::var("OPENAI_API_KEY")?,
    )))?;
    let model = provider.language_model("gpt-5.6-luna");

    let base = Request::builder()
        .reasoning(ReasoningConfig::Effort {
            effort: ReasoningEffort::Medium,
            // Ask for reasoning text or summaries where the provider offers them.
            output: Some(ReasoningOutput::Include),
        })
        .build();

    let mut messages = vec![Message::user(
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. \
         How much does the ball cost?",
    )];
    let mut request = base.clone();
    request.messages = messages.clone();
    let first = model.generate(request).await?;

    for part in first.reasoning() {
        let visible = part.visible_text();
        if visible.is_empty() {
            println!(
                "[reasoning kept for replay only: {} content item(s)]",
                part.content.len()
            );
        } else {
            println!("[reasoning] {visible}");
        }
    }
    println!("answer: {}", first.text());
    println!("reasoning tokens: {:?}", first.usage.reasoning_tokens);

    // Replay the whole assistant turn, then continue the conversation.
    messages.push(first.to_assistant_message());
    messages.push(Message::user(
        "Now double both prices. What does the bat cost?",
    ));
    let mut request = base;
    request.messages = messages;
    let second = model.generate(request).await?;
    println!("follow-up: {}", second.text());
    Ok(())
}
