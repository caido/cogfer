//! The OpenAI Chat Completions profile against any compatible server
//! (LiteLLM, Ollama, OpenRouter, vLLM, ...). Off OpenAI's own hosts the
//! profile sends only the portable wire spellings, so strict servers do not
//! reject unknown fields.
//!
//! ```sh
//! BASE_URL=http://localhost:11434/v1 MODEL=llama3.2 cargo run --example openai_compatible
//! BASE_URL=https://openrouter.ai/api/v1 API_KEY=sk-or-... MODEL=openai/gpt-5.6-luna \
//!     cargo run --example openai_compatible
//! ```

use caido_ai::transport::{HeaderName, HeaderValue};
use caido_ai::{Client, Credentials, Message, ProviderConfig, Request, StreamEvent};
use futures_util::StreamExt;
use url::Url;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url = Url::parse(&std::env::var("BASE_URL")?)?;
    let model_id = std::env::var("MODEL")?;
    // Local servers usually need no key.
    let credentials = match std::env::var("API_KEY") {
        Ok(key) => Credentials::api_key(key),
        Err(_) => Credentials::none(),
    };

    caido_ai::transport::install_default_crypto_provider();

    let client = Client::builder().build()?;
    let provider = client.provider(
        ProviderConfig::openai_chat(credentials)
            .with_base_url(base_url)
            // Headers some gateways read for attribution.
            .with_header(
                HeaderName::from_static("x-title"),
                HeaderValue::from_static("caido-ai example"),
            ),
    )?;
    let model = provider.language_model(model_id);

    let request = Request::builder()
        .message(Message::user("Say hello in five words or fewer."))
        .build();
    let mut stream = model.stream(request).await?;
    while let Some(event) = stream.next().await {
        match event {
            StreamEvent::StreamStart { warnings } => {
                // Settings the profile could not express for this server.
                for warning in warnings {
                    eprintln!("warning: {warning:?}");
                }
            }
            StreamEvent::TextDelta { delta, .. } => print!("{delta}"),
            StreamEvent::Finish { finish, usage } => {
                println!("\n\nfinish: {:?}\nusage:  {usage:?}", finish.reason);
            }
            _ => {}
        }
    }
    Ok(())
}
