use cogfer::ProviderConfig;

use super::cassette_suite;

cassette_suite! {
    config: ProviderConfig::openai_responses,
    live: |client| crate::common::provider_from_env_on(client, "OPENAI_API_KEY", ProviderConfig::openai_responses),
    model: "gpt-5.6-luna",
    scenarios: [
        generate_text,
        stream_text,
        stream_tool_loop,
        structured_output,
        reasoning_stream,
        unknown_model,
    ],
}
