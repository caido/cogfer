//! Anthropic on Amazon Bedrock: both credentials the service accepts, then a
//! streaming tool loop.
//!
//! With a Bedrock API key (a plain bearer token, no signing):
//!
//! ```sh
//! AWS_BEARER_TOKEN_BEDROCK=... cargo run --example bedrock
//! ```
//!
//! Or with IAM credentials, signed with SigV4:
//!
//! ```sh
//! AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1 \
//!   cargo run --example bedrock
//! ```

use std::sync::Arc;

use cogfer::aws::{AwsCredentials, SigV4Authenticator};
use cogfer::{
    Client, Credentials, Message, Provider, ProviderConfig, ReasoningConfig, ReasoningEffort,
    Request, StreamAccumulator, StreamEvent, ToolDefinition, ToolResultPart,
};
use futures_util::StreamExt;
use serde_json::json;

/// Bedrock serves Anthropic models through cross-region inference profiles, so
/// the id carries a `us.` prefix. A bare `anthropic.` id is rejected with
/// "on-demand throughput isn't supported".
const MODEL: &str = "us.anthropic.claude-sonnet-4-6";

/// Prefer the Bedrock API key when one is set, otherwise sign with SigV4.
fn provider(client: &Client, region: &str) -> Result<Provider, Box<dyn std::error::Error>> {
    if let Ok(key) = std::env::var("AWS_BEARER_TOKEN_BEDROCK") {
        eprintln!("[auth] Bedrock API key (Authorization: Bearer)");
        return Ok(client.provider(ProviderConfig::bedrock_anthropic(
            region,
            Credentials::bearer(key),
        )?)?);
    }

    eprintln!("[auth] SigV4");
    let credentials = AwsCredentials::new(
        std::env::var("AWS_ACCESS_KEY_ID")?,
        std::env::var("AWS_SECRET_ACCESS_KEY")?,
        std::env::var("AWS_SESSION_TOKEN").ok(),
        None,
        "environment",
    );
    Ok(client.provider(
        ProviderConfig::bedrock_anthropic(region, Credentials::none())?
            .with_authenticator(Arc::new(SigV4Authenticator::new(region, credentials))),
    )?)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    cogfer::transport::install_default_crypto_provider();
    let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".into());
    let client = Client::builder().build()?;
    let model = provider(&client, &region)?.language_model(MODEL);

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
