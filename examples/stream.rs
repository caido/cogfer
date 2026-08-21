//! Streaming: forward text deltas as they arrive while a
//! [`StreamAccumulator`] rebuilds the same [`caido_ai::GenerateResult`] a
//! blocking call would return.
//!
//! ```sh
//! OPENAI_API_KEY=... cargo run --example stream
//! ```

use caido_ai::{
    Client, Credentials, Message, ProviderConfig, Request, StreamAccumulator, StreamEvent,
};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    caido_ai::transport::install_default_crypto_provider();
    let client = Client::builder().build()?;
    let provider = client.provider(ProviderConfig::openai_responses(Credentials::api_key(
        std::env::var("OPENAI_API_KEY")?,
    )))?;
    let model = provider.language_model("gpt-5.6-luna");

    let request = Request::builder()
        .message(Message::user("Write a haiku about HTTP proxies."))
        .build();

    // Every stream starts with StreamStart and ends with exactly one Finish.
    // Failures after the stream is established arrive as an Error event
    // followed by a Finish with `FinishReason::Error`.
    let mut stream = model.stream(request).await?;
    let mut accumulator = StreamAccumulator::new();
    while let Some(event) = stream.next().await {
        match &event {
            StreamEvent::TextDelta { delta, .. } => print!("{delta}"),
            StreamEvent::Error { error } => eprintln!("\nstream error: {error}"),
            _ => {}
        }
        accumulator.push(event);
    }
    println!();

    // `into_result` returns the first stream error, if any.
    let result = accumulator.into_result()?;
    println!("finish: {:?}", result.finish.reason);
    println!("usage:  {:?}", result.usage);
    Ok(())
}
