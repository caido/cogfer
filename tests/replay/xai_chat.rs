//! xAI Chat Completions dialect. Recording needs `XAI_API_KEY` or the sign-in
//! cached by `cargo run --example xai_login`.

use caido_ai::ProviderConfig;

use super::cassette_suite;

cassette_suite! {
    config: ProviderConfig::xai_chat,
    live: |client| crate::common::xai_provider_on(client, ProviderConfig::xai_chat),
    model: "grok-4.3",
    scenarios: [
        generate_text,
        stream_text,
        stream_tool_loop,
        structured_output,
        reasoning_stream,
        unknown_model,
    ],
}
