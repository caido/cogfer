//! Live native-compaction validation. Ignored by default. Run explicitly with:
//!
//! ```sh
//! cargo test live_compaction:: -- --ignored --nocapture --test-threads=1
//! ```

mod support;

mod anthropic;
mod openai;
