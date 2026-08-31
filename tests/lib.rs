//! Integration tests. Mock and replay tests run offline. Live provider tests
//! and cassette recording are `#[ignore]`d and need API keys:
//!
//! ```sh
//! cargo test --test lib live:: -- --ignored --nocapture
//! cargo test --test lib live_compaction:: -- --ignored --nocapture --test-threads=1
//! cargo test --test lib replay::openai_responses::record -- --ignored --nocapture
//! ```

mod common;

mod anthropic;
#[cfg(feature = "aws")]
mod bedrock;
mod capabilities;
mod chatgpt;
mod core;
mod custom_urls;
mod gemini;
#[cfg(feature = "reqwest-transport")]
mod live;
#[cfg(feature = "reqwest-transport")]
mod live_compaction;
mod openai_chat;
mod openai_responses;
mod openrouter;
mod parity;
mod replay;
mod xai;
