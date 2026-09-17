//! ChatGPT subscription backend. Recording needs the sign-in cached by
//! `cargo run --example chatgpt_login` (or `CHATGPT_ACCESS_TOKEN`).

use cogfer::ProviderConfig;

use super::cassette_suite;

cassette_suite! {
    config: ProviderConfig::chatgpt,
    live: crate::common::chatgpt_provider_on,
    model: "gpt-5.6-sol",
    scenarios: [
        generate_text,
        stream_text,
        stream_tool_loop,
        structured_output,
        reasoning_stream,
        unknown_model,
    ],
}
