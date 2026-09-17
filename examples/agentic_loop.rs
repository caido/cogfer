//! A minimal streaming tool loop using OpenAI Responses.
//!
//! ```sh
//! OPENAI_API_KEY=sk-... cargo run --example agentic_loop
//! ```

use cogfer::{
    Client, Credentials, Message, ProviderConfig, ReasoningConfig, ReasoningEffort, Request,
    StreamAccumulator, StreamEvent, ToolDefinition, ToolResultPart,
};
use futures_util::StreamExt;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    cogfer::transport::install_default_crypto_provider();
    let client = Client::builder().build()?;
    let provider = client.provider(ProviderConfig::openai_responses(Credentials::api_key(
        std::env::var("OPENAI_API_KEY")?,
    )))?;
    let model = provider.language_model("gpt-5.6-sol");

    let base = Request::builder()
        .system("Use the get_current_time tool for time questions.")
        .message(Message::user("What time is it in Paris?"))
        .tool(ToolDefinition::new(
            "get_current_time",
            "Returns the current time for a city.",
            json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": false
            }),
        ))
        .reasoning(ReasoningConfig::effort(ReasoningEffort::Low))
        .max_output_tokens(2000)
        .build();

    let mut messages = base.messages.clone();
    loop {
        let mut request = base.clone();
        request.messages = messages.clone();

        let mut stream = model.stream(request).await?;
        let mut accumulator = StreamAccumulator::new();
        while let Some(event) = stream.next().await {
            match &event {
                StreamEvent::TextDelta { delta, .. } => print!("{delta}"),
                StreamEvent::ReasoningDelta { delta, .. } => eprint!("\x1b[2m{delta}\x1b[0m"),
                StreamEvent::ToolCall(call) => {
                    println!("\n[tool call] {}({})", call.name, call.arguments)
                }
                _ => {}
            }
            accumulator.push(event);
        }
        let result = accumulator.into_result()?;

        if !result.has_tool_calls() {
            println!("\n\nusage: {:?}", result.usage);
            break;
        }

        messages.push(result.to_assistant_message());
        for call in result.tool_calls() {
            messages.push(Message::tool_result(ToolResultPart::for_call(
                call,
                "It is 12:00 noon.",
            )));
        }
    }
    Ok(())
}
