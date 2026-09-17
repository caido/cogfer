use cogfer::{Compaction, Message, ProviderConfig, Request, StreamEvent};

use super::support::{SYSTEM, compaction_parts, oversized_history, print_cost, provider_from_env};
use crate::common::{assert_terminal_contract, collect, drain};

#[tokio::test]
#[ignore = "live"]
async fn openai_responses_compaction_triggers_and_replays() {
    let Some(provider) = provider_from_env("OPENAI_API_KEY", ProviderConfig::openai_responses)
    else {
        return;
    };
    let model = provider.language_model("gpt-5.6-luna");

    let mut request = Request::builder()
        .system(SYSTEM)
        .messages(oversized_history(6, 80))
        .max_output_tokens(2000)
        .build();
    request.compaction = Some(Compaction {
        trigger_input_tokens: Some(2000),
        ..Compaction::default()
    });
    let first = model.generate(request).await.expect("turn 1 succeeds");
    print_cost("openai-turn1", "gpt-5.6-luna", &first.usage);

    let parts = compaction_parts(&first);
    assert!(
        !parts.is_empty(),
        "expected a compaction item; finish: {:?}, warnings: {:?}",
        first.finish,
        first.warnings
    );
    assert!(
        parts[0].encrypted_content.is_some(),
        "openai compaction should carry encrypted_content"
    );
    let turn1_total = first.usage.total_input_tokens().expect("turn 1 usage");

    let mut messages = oversized_history(6, 80);
    messages.push(first.to_assistant_message());
    messages.push(Message::user("Reply with exactly one word: done"));
    let request = Request::builder()
        .messages(messages)
        .max_output_tokens(2000)
        .build();
    let second = model.generate(request).await.expect("full replay succeeds");
    print_cost("openai-turn2", "gpt-5.6-luna", &second.usage);

    // Opaque compaction restores server-side context, so input need not shrink.
    let pruned = vec![
        first.to_assistant_message(),
        Message::user("Reply with exactly one word: done"),
    ];
    let request = Request::builder()
        .messages(pruned)
        .max_output_tokens(2000)
        .build();
    let third = model
        .generate(request)
        .await
        .expect("pruned replay succeeds");
    print_cost("openai-turn3", "gpt-5.6-luna", &third.usage);
    let turn3_total = third.usage.total_input_tokens().expect("turn 3 usage");
    eprintln!("openai: input {turn1_total} -> {turn3_total} after pruned replay");
    assert!(!third.text().is_empty());
}

#[tokio::test]
#[ignore = "live"]
async fn openai_responses_compaction_streams() {
    let Some(provider) = provider_from_env("OPENAI_API_KEY", ProviderConfig::openai_responses)
    else {
        return;
    };
    let model = provider.language_model("gpt-5.6-sol");

    let mut request = Request::builder()
        .system(SYSTEM)
        .messages(oversized_history(6, 80))
        .max_output_tokens(2000)
        .build();
    request.compaction = Some(Compaction {
        trigger_input_tokens: Some(2000),
        ..Compaction::default()
    });
    let events = drain(model.stream(request).await.expect("stream establishes")).await;
    assert_terminal_contract(&events);
    let compactions: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Compaction(part) => Some(part),
            _ => None,
        })
        .collect();
    assert!(
        !compactions.is_empty(),
        "expected a streamed compaction event"
    );
    assert!(compactions[0].encrypted_content.is_some());
    if let Some(StreamEvent::Finish { usage, .. }) = events.last() {
        print_cost("openai-stream", "gpt-5.6-sol", usage);
    }
    let result = collect(&events).expect("stream result");
    assert!(!compaction_parts(&result).is_empty());
}
