//! `gpt-5.x` on Chat Completions rejects function tools unless a reasoning
//! effort is set explicitly (see [`ProviderConfig::openai_chat`]), so the tool
//! loop runs with `ReasoningConfig::Disabled` here.

use cogfer::ProviderConfig;

use super::cassette_suite;

cassette_suite! {
    config: ProviderConfig::openai_chat,
    live: |client| crate::common::provider_from_env_on(client, "OPENAI_API_KEY", ProviderConfig::openai_chat),
    model: "gpt-5.6-luna",
    scenarios: [
        generate_text,
        stream_text,
        stream_tool_loop_reasoning_disabled,
        structured_output,
        reasoning_stream,
        unknown_model,
    ],
}
