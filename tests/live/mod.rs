//! Live provider validation. Ignored by default. Run explicitly with:
//!
//! ```sh
//! cargo test live:: -- --ignored --nocapture
//! ```

mod anthropic;
mod openai;
mod openai_compatible;
mod openrouter;
mod subscriptions;
