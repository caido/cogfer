//! JSON-schema constrained output on OpenAI Chat Completions, parsed straight
//! into a serde type.
//!
//! ```sh
//! OPENAI_API_KEY=... cargo run --example structured_output
//! ```

use caido_ai::{Client, Credentials, Message, ProviderConfig, Request, StructuredOutput};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct Landmark {
    name: String,
    city: String,
    country: String,
    year_completed: u32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder().build()?;
    let provider = client.provider(ProviderConfig::openai_chat(Credentials::api_key(
        std::env::var("OPENAI_API_KEY")?,
    )))?;
    let model = provider.language_model("gpt-5.6-luna");

    // Strict mode (the default) asks the provider to guarantee the shape.
    // OpenAI's strict mode requires every property to be listed in
    // `required` and `additionalProperties: false` on every object.
    let request = Request::builder()
        .message(Message::user("Describe the Eiffel Tower."))
        .structured_output(StructuredOutput::new(
            "landmark",
            json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "city": {"type": "string"},
                    "country": {"type": "string"},
                    "year_completed": {"type": "integer"}
                },
                "required": ["name", "city", "country", "year_completed"],
                "additionalProperties": false
            }),
        ))
        .build();

    let result = model.generate(request).await?;
    let landmark: Landmark = result.structured_output()?;
    println!(
        "{} in {}, {} (completed {})",
        landmark.name, landmark.city, landmark.country, landmark.year_completed
    );
    Ok(())
}
