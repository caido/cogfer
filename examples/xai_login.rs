//! Sign in to SpaceXAI, persist the tokens, and test generation.
//!
//! ```sh
//! cargo run --example xai_login
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use cogfer::oauth::xai::{XaiAuthenticator, XaiOAuth};
use cogfer::{Client, Credentials, Message, ProviderConfig, Request};

#[path = "support/token_file.rs"]
mod token_file;
use token_file::TokenFile;

fn auth_file() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let base = std::env::var_os("APPDATA").map(PathBuf::from);
    #[cfg(not(target_os = "windows"))]
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")));
    base.map(|dir| dir.join("cogfer").join("xai.json"))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    cogfer::transport::install_default_crypto_provider();
    let export = std::env::args().any(|arg| arg == "--export");

    let oauth = XaiOAuth::with_default_transport()?;
    let device = oauth.start_device_authorization().await?;
    println!("Sign in with SpaceXAI:");
    match &device.verification_url_complete {
        Some(url) => println!("  1. Visit {url}"),
        None => println!("  1. Visit {}", device.verification_url),
    }
    println!("  2. Confirm code: {}", device.user_code);
    println!("Do not share this device code. Waiting for confirmation…");

    let tokens = oauth.wait_for_tokens(&device).await?;

    let token_store = match auth_file() {
        Some(path) => {
            let store = Arc::new(TokenFile::new(path));
            store.save(&tokens).await?;
            println!(
                "\nSigned in. Cached in {} (picked up by the live tests).",
                store.path().display()
            );
            Some(store)
        }
        None => {
            println!("\nSigned in, but no config directory was found; nothing cached.");
            None
        }
    };

    if export {
        println!("\nEnv exports (SECRETS: keep out of CI logs and scrollback):");
        println!("  export XAI_ACCESS_TOKEN='{}'", tokens.access_token);
        if let Some(refresh_token) = &tokens.refresh_token {
            println!("  export XAI_REFRESH_TOKEN='{refresh_token}'");
        }
    } else {
        println!("(Pass --export to print XAI_* env exports.)");
    }

    let mut authenticator = XaiAuthenticator::with_default_transport(tokens)?;
    if let Some(store) = token_store {
        authenticator = authenticator.with_token_store(store);
    }
    let client = Client::builder().build()?;
    let provider = client.provider(
        ProviderConfig::xai(Credentials::none()).with_authenticator(Arc::new(authenticator)),
    )?;
    let result = provider
        .language_model("grok-4.6")
        .generate(
            Request::builder()
                .message(Message::user("Say OK if you can read this."))
                .build(),
        )
        .await?;
    println!("\nBackend answered: {}", result.text());
    Ok(())
}
