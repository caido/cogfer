//! SpaceXAI Responses dialect. Recording needs `XAI_API_KEY` or the sign-in
//! cached by `cargo run --example xai_login`.

use cogfer::ProviderConfig;

use super::cassette_suite;

cassette_suite! {
    config: ProviderConfig::xai,
    live: |client| crate::common::xai_provider_on(client, ProviderConfig::xai),
    model: "grok-4.5",
    scenarios: [
        generate_text,
        stream_text,
        stream_tool_loop,
        structured_output,
        reasoning_stream,
        unknown_model,
    ],
}
