# caido-ai

A library for calling LLMs through one typed request model, stream contract, and error taxonomy. Supports OpenAI (Responses and Chat Completions), ChatGPT subscriptions, xAI, OpenRouter, Anthropic, and Gemini.

## Quick start

```toml
[dependencies]
caido-ai = { git = "https://github.com/caido/caido-ai" }
```

```rust
use caido_ai::{Client, Credentials, Message, ProviderConfig, Request};

// Once per process, unless the host already installed a Rustls provider.
caido_ai::transport::install_default_crypto_provider();
let client = Client::builder().build()?;
let provider = client.provider(ProviderConfig::openai_responses(
    Credentials::api_key(std::env::var("OPENAI_API_KEY")?),
))?;
let model = provider.language_model("gpt-5.6-luna");

let request = Request::builder()
    .message(Message::user("Why is the sky blue?"))
    .build();
println!("{}", model.generate(request).await?.text());
```

Streaming:

```rust
use futures_util::StreamExt;

let mut stream = model.stream(request).await?;
while let Some(event) = stream.next().await {
    if let caido_ai::StreamEvent::TextDelta { delta, .. } = event {
        print!("{delta}");
    }
}
```

Events are designed for agentic environments and map cleanly onto AI SDK stream parts. A typical event sequence:

```
StreamStart { warnings: [] }
ResponseMetadata { id: "resp_abc123", model: "gpt-5.6-luna", .. }
ReasoningStart { id: "rs_0" }
ReasoningDelta { id: "rs_0", delta: "The user asks why the sky is blue" }
ReasoningDelta { id: "rs_0", delta: " — Rayleigh scattering explains it." }
ReasoningEnd { id: "rs_0", part: ReasoningPart { .. } }
TextStart { id: "0", .. }
TextDelta { id: "0", delta: "The sky is blue" }
TextDelta { id: "0", delta: " because sunlight scatters" }
TextDelta { id: "0", delta: " off air molecules." }
TextEnd { id: "0", .. }
Finish { reason: Stop, usage: Usage { input_tokens: 13, output_tokens: 12, .. }, .. }
```

## Features

- Blocking and streaming completions
- Tool calls and reasoning, replayable across turns
- Structured output, context compaction
- Normalized errors, usage, and warnings
- ChatGPT and xAI subscription OAuth
- Pluggable HTTP transport

## Examples and tests

```sh
OPENAI_API_KEY=... cargo run --example stream
cargo test   # no network tests
OPENAI_API_KEY=... cargo test --test lib live::openai -- --ignored --nocapture # will use OPENAI_API_KEY and call openai for tests
```
