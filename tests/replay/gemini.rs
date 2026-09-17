use cogfer::ProviderConfig;

use super::cassette_suite;

cassette_suite! {
    config: ProviderConfig::gemini,
    live: |client| crate::common::provider_from_env_on(client, "GOOGLE_API_KEY", ProviderConfig::gemini),
    model: "gemini-3.6-flash",
    scenarios: [
        generate_text,
        stream_text,
        stream_tool_loop,
        structured_output,
        reasoning_stream,
        unknown_model,
    ],
}
