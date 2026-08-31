use llmwire::ProviderConfig;

use super::cassette_suite;

cassette_suite! {
    config: ProviderConfig::openrouter,
    live: |client| crate::common::provider_from_env_on(client, "OPENROUTER_API_KEY", ProviderConfig::openrouter),
    model: "openai/gpt-5.6-luna",
    scenarios: [
        generate_text,
        stream_text,
        stream_tool_loop,
        structured_output,
        reasoning_stream,
        unknown_model,
    ],
}
