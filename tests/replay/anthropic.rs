use llmwire::ProviderConfig;

use super::cassette_suite;

cassette_suite! {
    config: ProviderConfig::anthropic,
    live: |client| crate::common::provider_from_env_on(client, "ANTHROPIC_API_KEY", ProviderConfig::anthropic),
    model: "claude-sonnet-4-6",
    scenarios: [
        generate_text,
        stream_text,
        stream_tool_loop,
        structured_output,
        reasoning_stream,
        unknown_model,
    ],
}
