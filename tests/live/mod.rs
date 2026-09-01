//! Live provider validation. Ignored by default. Run explicitly with:
//!
//! ```sh
//! cargo test live:: -- --ignored --nocapture
//! ```

mod anthropic;
#[cfg(feature = "aws")]
pub(crate) mod bedrock;
mod bedrock_openai;
mod openai;
mod openai_compatible;
mod openrouter;
mod subscriptions;
