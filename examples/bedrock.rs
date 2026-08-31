//! Anthropic on Amazon Bedrock, signed with SigV4 from environment credentials.
//!
//! ```sh
//! AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1 \
//!   cargo run --example bedrock
//! ```
//!
//! With a Bedrock API key instead, use `Credentials::bearer(key)` and skip
//! the authenticator.

use std::sync::Arc;

use llmwire::aws::{AwsCredentials, SigV4Authenticator};
use llmwire::{Client, Credentials, Message, ProviderConfig, Request};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    llmwire::transport::install_default_crypto_provider();
    let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".into());
    let mut credentials = AwsCredentials::new(
        std::env::var("AWS_ACCESS_KEY_ID")?,
        std::env::var("AWS_SECRET_ACCESS_KEY")?,
    );
    if let Ok(token) = std::env::var("AWS_SESSION_TOKEN") {
        credentials = credentials.with_session_token(token);
    }

    let client = Client::builder().build()?;
    let provider = client.provider(
        ProviderConfig::bedrock_anthropic(&region, Credentials::none())?
            .with_authenticator(Arc::new(SigV4Authenticator::new(&region, credentials))),
    )?;
    let model = provider.language_model("anthropic.claude-sonnet-4-5-20250929-v1:0");

    let request = Request::builder()
        .system("Answer in one short sentence.")
        .message(Message::user("Why is the sky blue?"))
        .build();
    let result = model.generate(request).await?;
    println!("{}", result.text());
    Ok(())
}
