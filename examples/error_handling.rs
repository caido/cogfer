//! What a failed call looks like: one normalized [`cogfer::Error`] with a
//! stable [`ErrorKind`], the provider's status/code/request id, and a retry
//! hint, so hosts can map it to their own responses and retry policy.
//!
//! ```sh
//! OPENAI_API_KEY=... cargo run --example error_handling
//! ```

use cogfer::{Client, Credentials, ErrorKind, Message, ProviderConfig, Request};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    cogfer::transport::install_default_crypto_provider();
    let client = Client::builder().build()?;
    let provider = client.provider(ProviderConfig::openai_responses(Credentials::api_key(
        std::env::var("OPENAI_API_KEY")?,
    )))?;

    let request = Request::builder().message(Message::user("hi")).build();
    let Err(error) = provider
        .language_model("this-model-does-not-exist")
        .generate(request)
        .await
    else {
        println!("unexpectedly succeeded");
        return Ok(());
    };

    // `Display` is credential-free and safe to log or show to users.
    println!("{error}");
    println!();
    println!("kind:        {:?}", error.kind());
    println!("status:      {:?}", error.status());
    println!("code:        {:?}", error.code());
    println!("request id:  {:?}", error.request_id());
    println!("retryable:   {}", error.retryable());
    println!("retry after: {:?}", error.retry_after());

    // Hosts branch on the kind, never on provider message text.
    let advice = match error.kind() {
        ErrorKind::NotFound | ErrorKind::InvalidRequest => "fix the request or model id",
        ErrorKind::Authentication | ErrorKind::Permission => "check the configured credentials",
        ErrorKind::RateLimited | ErrorKind::Overloaded => "back off, then retry",
        ErrorKind::ContextLength => "shorten the prompt or enable compaction",
        _ if error.retryable() => "retry with backoff",
        _ => "report to the user",
    };
    println!("advice:      {advice}");
    Ok(())
}
