use super::cassette_suite;
use crate::common::bedrock_config;

cassette_suite! {
    config: bedrock_config,
    live: crate::common::bedrock_provider_on,
    // Bedrock serves Anthropic models only through cross-region inference
    // profiles, so the model id carries the `us.` prefix rather than being the
    // bare `anthropic.` id.
    model: "us.anthropic.claude-sonnet-4-6",
    scenarios: [
        generate_text,
        stream_text,
        stream_tool_loop,
        structured_output,
        reasoning_stream,
        unknown_model,
    ],
}
