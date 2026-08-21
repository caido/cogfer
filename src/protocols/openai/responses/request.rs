use std::collections::{BTreeSet, HashMap};

use serde_json::{Value, json};

use super::ResponsesDialect;
use crate::error::{Error, ErrorKind, Result};
use crate::message::{AssistantPart, Message, ReasoningContent, UserPart};
use crate::protocols::openai::shared::{ToolShape, insert_tools, json_schema_format};
use crate::protocols::{ApiProfile, ProtocolContext, foreign_origin};
use crate::request::{ReasoningConfig, ReasoningOutput};
use crate::response::Warning;

/// Lower the message history (without the system prompt, which uses
/// `instructions` or [`system_item`]) into Responses `input` items.
///
/// Opaque provider state in the history (reasoning items, encrypted content,
/// item ids, server tool payloads) is replayed only to the
/// [`origin_metadata`](crate::protocols::origin_metadata)-stamped profile that
/// produced it; other backends cannot verify it and reject the request.
fn lower_input(ctx: &ProtocolContext<'_>, warnings: &mut Vec<Warning>) -> Result<Value> {
    let mut lowering = InputLowering {
        profile: ctx.profile,
        items: Vec::new(),
        foreign_origins: BTreeSet::new(),
        warnings,
    };
    let mut call_index = HashMap::new();
    for message in &ctx.request.messages {
        match message {
            Message::User { content } => {
                let mut parts = Vec::new();
                for part in content {
                    let UserPart::Text { text } = part;
                    parts.push(json!({"type": "input_text", "text": text}));
                }
                lowering
                    .items
                    .push(json!({"type": "message", "role": "user", "content": parts}));
            }
            Message::Assistant {
                content,
                provider_metadata,
            } => {
                let origin = foreign_origin(provider_metadata, ctx.profile);
                for part in content {
                    if let AssistantPart::ToolCall(call) = part {
                        call_index.insert(
                            call.call_id.as_str(),
                            call.provider_call_id.as_deref().unwrap_or(&call.call_id),
                        );
                    }
                    lowering.lower_assistant_part(part, origin)?;
                }
            }
            Message::Tool { content } => {
                for result in content {
                    let call_id = result
                        .provider_call_id
                        .as_deref()
                        .or_else(|| call_index.get(result.call_id.as_str()).copied())
                        .unwrap_or(&result.call_id);
                    lowering.items.push(json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result.content.to_text(),
                    }));
                }
            }
        }
    }
    for origin in &lowering.foreign_origins {
        lowering.warnings.push(Warning::other(format!(
            "dropped opaque reasoning or tool items produced by `{origin}`; `{}` cannot verify \
             state from another provider",
            ctx.profile.as_str()
        )));
    }
    Ok(Value::Array(lowering.items))
}

/// The system prompt as a leading `input` item.
pub(crate) fn system_item(system: &str) -> Value {
    json!({
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": system}],
    })
}

/// History lowering state: emitted items plus the foreign profiles whose
/// opaque items were dropped, for one warning each.
struct InputLowering<'a> {
    profile: ApiProfile,
    items: Vec<Value>,
    foreign_origins: BTreeSet<String>,
    warnings: &'a mut Vec<Warning>,
}

impl InputLowering<'_> {
    /// Lower one assistant part. `origin` names the part's producing profile
    /// when it differs from the serving profile.
    fn lower_assistant_part(&mut self, part: &AssistantPart, origin: Option<&str>) -> Result<()> {
        match part {
            AssistantPart::Text {
                text,
                provider_metadata,
            } => {
                let namespace = provider_metadata.get("openai");
                let is_refusal = namespace
                    .and_then(|openai| openai.get("content_type"))
                    .and_then(Value::as_str)
                    == Some("refusal");
                let content = if is_refusal {
                    json!([{"type": "refusal", "refusal": text}])
                } else {
                    json!([{"type": "output_text", "text": text}])
                };
                let mut item = json!({
                    "type": "message",
                    "role": "assistant",
                    "content": content,
                });
                // The phase marker only round-trips to the backend that emitted it.
                if origin.is_none()
                    && let Some(phase) = namespace
                        .and_then(|openai| openai.get("phase"))
                        .and_then(Value::as_str)
                {
                    item["phase"] = json!(phase);
                }
                self.items.push(item);
            }
            AssistantPart::Reasoning(reasoning) => {
                // OpenRouter reasoning is recognizable by its verbatim
                // `reasoning_details` metadata even on an unstamped turn.
                let origin = origin.or_else(|| {
                    reasoning
                        .provider_metadata
                        .get(ApiProfile::OpenRouter.namespace())
                        .map(|_| ApiProfile::OpenRouter.as_str())
                });
                if let Some(origin) = origin {
                    let replayable = reasoning.id.is_some()
                        || reasoning
                            .content
                            .iter()
                            .any(|content| matches!(content, ReasoningContent::Encrypted { .. }));
                    if replayable {
                        self.foreign_origins.insert(origin.to_string());
                    }
                    return Ok(());
                }
                let mut summary = Vec::new();
                let mut texts = Vec::new();
                let mut encrypted = None;
                for content in &reasoning.content {
                    match content {
                        ReasoningContent::Summary { text } => {
                            summary.push(json!({"type": "summary_text", "text": text}));
                        }
                        ReasoningContent::Text { text, .. } => {
                            texts.push(json!({"type": "reasoning_text", "text": text}));
                        }
                        ReasoningContent::Encrypted { data } => encrypted = Some(data.clone()),
                        ReasoningContent::Redacted { .. } => {}
                    }
                }
                if reasoning.id.is_none() && encrypted.is_none() {
                    return Ok(());
                }
                let mut item = json!({
                    "type": "reasoning",
                    "summary": summary,
                });
                if let Some(id) = &reasoning.id {
                    item["id"] = json!(id);
                }
                if !texts.is_empty() {
                    item["content"] = Value::Array(texts);
                }
                if let Some(encrypted) = encrypted {
                    item["encrypted_content"] = json!(encrypted);
                }
                self.items.push(item);
            }
            AssistantPart::ToolCall(call) => {
                let mut item = json!({
                    "type": "function_call",
                    "call_id": call.provider_call_id.as_deref().unwrap_or(&call.call_id),
                    "name": call.name,
                    "arguments": call.arguments,
                });
                // A foreign item id is omitted without a warning: the call
                // itself still replays through `call_id`, so nothing is lost.
                if origin.is_none()
                    && let Some(item_id) = &call.item_id
                {
                    item["id"] = json!(item_id);
                }
                self.items.push(item);
            }
            AssistantPart::Compaction(compaction) => {
                if let Some(encrypted) = &compaction.encrypted_content {
                    if let Some(origin) = origin {
                        // Skipping would silently lose the history the item replaced.
                        return Err(Error::new(
                            ErrorKind::UnsupportedContent,
                            format!(
                                "history contains an opaque compaction item from `{origin}`; \
                                 `{}` cannot replay it",
                                self.profile.as_str()
                            ),
                        ));
                    }
                    let mut item = json!({
                        "type": "compaction",
                        "encrypted_content": encrypted,
                    });
                    if let Some(id) = &compaction.id {
                        item["id"] = json!(id);
                    }
                    self.items.push(item);
                } else if let Some(summary) = &compaction.content {
                    // Anthropic-style summary compaction has no native item
                    // here; keep the summarized history as plain text.
                    self.items.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": summary}],
                    }));
                    self.warnings.push(Warning::other(
                        "compaction summary was flattened to plain assistant text: this protocol \
                         replays only encrypted compaction items",
                    ));
                }
            }
            AssistantPart::ProviderTool {
                provider_tool: tool,
            } => {
                if tool.namespace != "openai" {
                    return Ok(());
                }
                if let Some(origin) = origin {
                    self.foreign_origins.insert(origin.to_string());
                    return Ok(());
                }
                self.items.push(tool.payload.clone());
            }
        }
        Ok(())
    }
}

fn insert_reasoning_configuration(
    ctx: &ProtocolContext<'_>,
    object: &mut serde_json::Map<String, Value>,
    warnings: &mut Vec<Warning>,
) {
    let Some(reasoning) = &ctx.request.reasoning else {
        return;
    };
    let mut config = serde_json::Map::new();
    let output = match reasoning {
        ReasoningConfig::Disabled => {
            config.insert("effort".into(), json!("none"));
            None
        }
        ReasoningConfig::Effort { effort, output } => {
            config.insert("effort".into(), json!(effort.as_str()));
            *output
        }
        ReasoningConfig::Budget { output, .. } => {
            warnings.push(Warning::unsupported_setting(
                "reasoning.budget",
                "openai-responses uses discrete reasoning efforts, not token budgets",
            ));
            *output
        }
    };
    if output == Some(ReasoningOutput::Include) {
        config.insert("summary".into(), json!("auto"));
    }
    if !config.is_empty() {
        object.insert("reasoning".into(), Value::Object(config));
    }
}

fn insert_compaction_configuration(
    ctx: &ProtocolContext<'_>,
    object: &mut serde_json::Map<String, Value>,
    warnings: &mut Vec<Warning>,
) {
    let Some(compaction) = &ctx.request.compaction else {
        return;
    };
    let mut entry = serde_json::Map::new();
    entry.insert("type".into(), json!("compaction"));
    if let Some(threshold) = compaction.trigger_input_tokens {
        let clamped = threshold.max(1000);
        if clamped != threshold {
            warnings.push(Warning::other(format!(
                "compact_threshold raised to the API minimum 1000 (requested {threshold})",
            )));
        }
        entry.insert("compact_threshold".into(), json!(clamped));
    }
    if compaction.pause_after_compaction == Some(true) {
        warnings.push(Warning::unsupported_setting(
            "compaction.pause_after_compaction",
            "openai-responses compaction cannot pause the turn",
        ));
    }
    if compaction.instructions.is_some() {
        warnings.push(Warning::unsupported_setting(
            "compaction.instructions",
            "openai-responses compaction does not accept custom instructions",
        ));
    }
    object.insert(
        "context_management".into(),
        Value::Array(vec![Value::Object(entry)]),
    );
}

fn insert_sampling_configuration(
    ctx: &ProtocolContext<'_>,
    chatgpt: bool,
    object: &mut serde_json::Map<String, Value>,
    warnings: &mut Vec<Warning>,
) {
    let request = ctx.request;
    if chatgpt {
        for (setting, present) in [
            ("max_output_tokens", request.max_output_tokens.is_some()),
            ("temperature", request.temperature.is_some()),
            ("top_p", request.top_p.is_some()),
        ] {
            if present {
                warnings.push(Warning::unsupported_setting(
                    setting,
                    format!("the chatgpt backend does not accept `{setting}`"),
                ));
            }
        }
    } else {
        if let Some(max) = request.max_output_tokens {
            object.insert("max_output_tokens".into(), json!(max));
        }
        if let Some(temperature) = request.temperature {
            object.insert("temperature".into(), json!(temperature));
        }
        if let Some(top_p) = request.top_p {
            object.insert("top_p".into(), json!(top_p));
        }
    }
    for (setting, present) in [
        ("top_k", request.top_k.is_some()),
        ("stop_sequences", !request.stop_sequences.is_empty()),
        ("presence_penalty", request.presence_penalty.is_some()),
        ("frequency_penalty", request.frequency_penalty.is_some()),
        ("seed", request.seed.is_some()),
    ] {
        if present {
            warnings.push(Warning::unsupported_setting(
                setting,
                format!("openai-responses does not support `{setting}`"),
            ));
        }
    }
}

fn insert_streaming_configuration(
    object: &mut serde_json::Map<String, Value>,
    streaming: bool,
    chatgpt: bool,
) {
    if chatgpt {
        object.insert("stream".into(), json!(true));
    } else if streaming {
        object.insert("stream".into(), json!(true));
        object.insert(
            "stream_options".into(),
            json!({"include_obfuscation": false}),
        );
    }
}

/// Build the Responses request body plus lowering warnings.
pub(crate) fn lower_body(
    ctx: &ProtocolContext<'_>,
    streaming: bool,
    dialect: ResponsesDialect,
) -> Result<(Value, Vec<Warning>)> {
    let mut warnings = Vec::new();
    let request = ctx.request;
    let chatgpt = dialect.is_chatgpt();
    let mut input = lower_input(ctx, &mut warnings)?;
    let system = request.system_prompt();
    if let (Some(system), false) = (system, chatgpt) {
        let items = input.as_array_mut().expect("input is an array");
        items.insert(0, system_item(system));
    }

    let mut body = json!({
        "model": ctx.model,
        "input": input,
        "store": false,
        "include": ["reasoning.encrypted_content"],
    });
    let object = body.as_object_mut().expect("body is an object");

    if let (Some(system), true) = (system, chatgpt) {
        object.insert("instructions".into(), json!(system));
    }

    insert_tools(request, object, ToolShape::Flat);
    if let Some(output) = &request.structured_output {
        let mut format = json_schema_format(output);
        format["type"] = json!("json_schema");
        object.insert("text".into(), json!({"format": format}));
    }
    insert_reasoning_configuration(ctx, object, &mut warnings);
    insert_compaction_configuration(ctx, object, &mut warnings);
    insert_sampling_configuration(ctx, chatgpt, object, &mut warnings);
    insert_streaming_configuration(object, streaming, chatgpt);

    if let Some(options) = request.provider_options.get("openai") {
        crate::util::json_merge(&mut body, options.clone());
    }

    Ok((body, warnings))
}
