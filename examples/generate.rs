//! The smallest possible call: one blocking completion against Gemini, then
//! the normalized result (text, finish reason, usage, response identity).
//!
//! ```sh
//! GOOGLE_API_KEY=... cargo run --example generate
//! ```

use llmwire::{Client, Credentials, Message, ProviderConfig, Request};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    llmwire::transport::install_default_crypto_provider();
    let client = Client::builder().build()?;
    let provider = client.provider(ProviderConfig::gemini(Credentials::api_key(std::env::var(
        "GOOGLE_API_KEY",
    )?)))?;
    let model = provider.language_model("gemini-3.6-flash");

    let request = Request::builder()
        .system("Answer in one short sentence.")
        .message(Message::user("Why is the sky blue?"))
        .build();
    let result = model.generate(request).await?;

    println!("{}", result.text());
    println!();
    println!("finish:   {:?}", result.finish.reason);
    println!("usage:    {:?}", result.usage);
    println!("response: {:?}", result.response);
    for warning in &result.warnings {
        println!("warning:  {warning:?}");
    }
    Ok(())
}
