use cogfer::{AssistantPart, Message, Usage};

pub(crate) use crate::common::provider_from_env;

pub(crate) fn print_cost(label: &str, model: &str, usage: &Usage) {
    eprintln!(
        "COST label={label} model={model} input={} cached={} cache_write={} output={} reasoning={}",
        usage.input_tokens.unwrap_or(0),
        usage.cached_input_tokens.unwrap_or(0),
        usage.cache_creation_input_tokens.unwrap_or(0),
        usage.output_tokens.unwrap_or(0),
        usage.reasoning_tokens.unwrap_or(0),
    );
}

pub(crate) const SYSTEM: &str = "You are a terse assistant for a garden club journal.";

fn filler_text(turn: usize, sentences: usize) -> String {
    let mut out = String::with_capacity(sentences * 160);
    for s in 0..sentences {
        out.push_str(&format!(
            "Entry {turn}-{s}: on day {} the walking club visited the botanical gardens near \
             mile marker {}, counted {} tulip varieties in bloom, and rated the afternoon \
             picnic {} out of ten. ",
            s + turn,
            s * 7 + turn,
            (s * 31 + turn * 17) % 500,
            (s * 13 + 5) % 10,
        ));
    }
    out
}

pub(crate) fn oversized_history(turns: usize, sentences_per_turn: usize) -> Vec<Message> {
    let mut messages = Vec::new();
    for turn in 0..turns {
        messages.push(Message::user(filler_text(turn, sentences_per_turn)));
        messages.push(Message::assistant(format!(
            "Noted outing {turn}; lovely weather throughout."
        )));
    }
    messages.push(Message::user(
        "Reply with exactly one word: acknowledged".to_string(),
    ));
    messages
}

pub(crate) fn compaction_parts(result: &cogfer::GenerateResult) -> Vec<&cogfer::CompactionPart> {
    result
        .content
        .iter()
        .filter_map(|part| match part {
            AssistantPart::Compaction(part) => Some(part),
            _ => None,
        })
        .collect()
}
