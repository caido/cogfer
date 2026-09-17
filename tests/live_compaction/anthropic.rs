use cogfer::{Compaction, FinishReason, Message, ProviderConfig, Request, StreamEvent};

use super::support::{SYSTEM, compaction_parts, oversized_history, print_cost, provider_from_env};
use crate::common::{assert_terminal_contract, collect, drain};

#[tokio::test]
#[ignore = "live"]
async fn anthropic_compaction_triggers_and_replays() {
    let Some(provider) = provider_from_env("ANTHROPIC_API_KEY", ProviderConfig::anthropic) else {
        return;
    };
    let model = provider.language_model("claude-opus-5");

    let mut request = Request::builder()
        .system(SYSTEM)
        .messages(oversized_history(10, 100))
        .max_output_tokens(4000)
        .build();
    request.compaction = Some(Compaction {
        trigger_input_tokens: Some(50_000),
        ..Compaction::default()
    });
    let first = model.generate(request).await.expect("turn 1 succeeds");
    print_cost("anthropic-turn1", "claude-opus-5", &first.usage);
    eprintln!("anthropic: turn 1 finish={:?}", first.finish);

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
        "anthropic compaction should carry a readable summary (finish {:?})",
        first.finish
    );
    eprintln!("anthropic: compaction summary is {} chars", summary.len());
    let turn1_input = first.usage.total_input_tokens().expect("turn 1 usage");
    assert!(
        turn1_input > 50_000,
        "summed multi-pass usage should include the compaction pass ({turn1_input})"
    );

    let resume = vec![
        first.to_assistant_message(),
        Message::user("Reply with exactly one word: done"),
    ];
    let request = Request::builder()
        .system(SYSTEM)
        .messages(resume)
        .max_output_tokens(300)
        .build();
    let second = model
        .generate(request)
        .await
        .expect("pruned replay succeeds");
    print_cost("anthropic-turn2", "claude-opus-5", &second.usage);
    let turn2_input = second.usage.total_input_tokens().expect("turn 2 usage");
    eprintln!("anthropic: input {turn1_input} -> {turn2_input} after pruned replay");
    assert!(
        turn2_input < turn1_input / 4,
        "pruned compacted history should shrink billed input ({turn1_input} -> {turn2_input})"
    );
    assert!(!second.text().is_empty());
}

#[tokio::test]
#[ignore = "live"]
async fn anthropic_compaction_pause_and_pruned_resume() {
    let Some(provider) = provider_from_env("ANTHROPIC_API_KEY", ProviderConfig::anthropic) else {
        return;
    };
    let model = provider.language_model("claude-opus-5");

    let mut request = Request::builder()
        .system(SYSTEM)
        .messages(oversized_history(10, 100))
        .max_output_tokens(4000)
        .build();
    request.compaction = Some(Compaction {
        trigger_input_tokens: Some(50_000),
        pause_after_compaction: Some(true),
        instructions: Some(
            "Summarize the outings reviewed so far, keeping every outing number.".to_string(),
        ),
    });
    let paused = model.generate(request).await.expect("paused turn succeeds");
    print_cost("anthropic-pause", "claude-opus-5", &paused.usage);
    assert_eq!(
        paused.finish.reason,
        FinishReason::Paused,
        "pause_after_compaction should pause the turn (raw: {:?})",
        paused.finish.raw
    );
    let parts = compaction_parts(&paused);
    assert!(!parts.is_empty(), "paused turn must carry the compaction");
    assert!(
        parts[0].content.as_deref().is_some_and(|s| !s.is_empty()),
        "paused compaction must carry the summary"
    );
    assert!(
        paused.usage.total_input_tokens().unwrap_or(0) > 50_000,
        "paused-turn usage must be recovered from iterations: {:?}",
        paused.usage
    );

    let resume = vec![
        paused.to_assistant_message(),
        Message::user("Reply with exactly one word: resumed"),
    ];
    let request = Request::builder()
        .system(SYSTEM)
        .messages(resume)
        .max_output_tokens(300)
        .build();
    let resumed = model
        .generate(request)
        .await
        .expect("pruned resume succeeds");
    print_cost("anthropic-resume", "claude-opus-5", &resumed.usage);
    assert!(!resumed.text().is_empty());
    let resume_input = resumed.usage.total_input_tokens().unwrap_or(0);
    assert!(
        resume_input > 100,
        "resume should replay the compaction summary (input {resume_input})"
    );
}

#[tokio::test]
#[ignore = "live"]
async fn anthropic_compaction_streams() {
    let Some(provider) = provider_from_env("ANTHROPIC_API_KEY", ProviderConfig::anthropic) else {
        return;
    };
    let model = provider.language_model("claude-opus-5");

    let mut request = Request::builder()
        .system(SYSTEM)
        .messages(oversized_history(10, 100))
        .max_output_tokens(4000)
        .build();
    request.compaction = Some(Compaction {
        trigger_input_tokens: Some(50_000),
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
    assert!(
        compactions[0]
            .content
            .as_deref()
            .is_some_and(|s| !s.is_empty()),
        "streamed compaction must carry the summary"
    );
    if let Some(StreamEvent::Finish { usage, finish }) = events.last() {
        print_cost("anthropic-stream", "claude-opus-5", usage);
        eprintln!("anthropic-stream: finish={finish:?}");
    }
    let result = collect(&events).expect("stream result");
    assert!(!compaction_parts(&result).is_empty());
}
