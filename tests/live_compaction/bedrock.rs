//! Native compaction on Amazon Bedrock, over both credentials the service
//! accepts. Bedrock opts into the beta through an `anthropic_beta` body field
//! rather than the direct API's `anthropic-beta` header.
//!
//! Bedrock rejects a `trigger.value` below 50000, so the history has to be
//! large enough to cross that floor: 10 turns bills ~47k tokens, 14 clears it.

use llmwire::{Compaction, Message, Provider, Request};

use super::support::{SYSTEM, compaction_parts, oversized_history, print_cost};
use crate::live::bedrock::{MODEL, bearer_provider, sigv4_provider};

/// Trigger compaction on an oversized history, then replay the pruned turn so
/// the summary alone carries the context forward.
async fn compaction_triggers_and_replays(provider: Provider, label: &str) {
    let model = provider.language_model(MODEL);

    let mut request = Request::builder()
        .system(SYSTEM)
        .messages(oversized_history(14, 100))
        .max_output_tokens(4000)
        .build();
    request.compaction = Some(Compaction {
        trigger_input_tokens: Some(50_000),
        ..Compaction::default()
    });
    let first = model.generate(request).await.expect("turn 1 succeeds");
    print_cost(&format!("{label}-turn1"), MODEL, &first.usage);

    let parts = compaction_parts(&first);
    assert!(
        !parts.is_empty(),
        "expected a compaction block; finish: {:?}, metadata: {:?}",
        first.finish,
        first.provider_metadata
    );
    let summary = parts[0].content.as_deref().unwrap_or_default();
    assert!(
        !summary.is_empty(),
        "bedrock compaction should carry a readable summary (finish {:?})",
        first.finish
    );
    let turn1_input = first.usage.total_input_tokens().expect("turn 1 usage");
    assert!(
        turn1_input > 50_000,
        "summed multi-pass usage should include the compaction pass ({turn1_input})"
    );

    let request = Request::builder()
        .system(SYSTEM)
        .messages(vec![
            first.to_assistant_message(),
            Message::user("Reply with exactly one word: done"),
        ])
        .max_output_tokens(300)
        .build();
    let second = model
        .generate(request)
        .await
        .expect("pruned replay succeeds");
    print_cost(&format!("{label}-turn2"), MODEL, &second.usage);
    let turn2_input = second.usage.total_input_tokens().expect("turn 2 usage");
    assert!(
        turn2_input < turn1_input / 4,
        "pruned compacted history should shrink billed input ({turn1_input} -> {turn2_input})"
    );
    assert!(!second.text().is_empty());
}

#[tokio::test]
#[ignore = "live"]
async fn bedrock_bearer_compaction_triggers_and_replays() {
    let Some(provider) = bearer_provider() else {
        return;
    };
    compaction_triggers_and_replays(provider, "bedrock-bearer").await;
}

#[tokio::test]
#[ignore = "live"]
async fn bedrock_sigv4_compaction_triggers_and_replays() {
    let Some(provider) = sigv4_provider() else {
        return;
    };
    compaction_triggers_and_replays(provider, "bedrock-sigv4").await;
}
