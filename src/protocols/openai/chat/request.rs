use serde_json::{Map, Value, json};

use super::ChatDialect;
use crate::error::Result;
use crate::http::join_url;
use crate::message::{AssistantPart, Message, ReasoningContent, ReasoningPart, UserPart};
use crate::protocols::openai::shared::{ToolShape, insert_tools, json_schema_format};
use crate::protocols::{
    LoweredRequest, ProtocolContext, ResolvedReasoning, foreign_origin, resolve_reasoning,
};
use crate::request::{ReasoningOutput, Request};
use crate::response::Warning;
use crate::transport::HttpRequest;

pub(crate) fn lower_chat(
    ctx: &ProtocolContext<'_>,
    streaming: bool,
    declared: ChatDialect,
) -> Result<LoweredRequest> {
    let mut warnings = Vec::new();
    let request = ctx.request;
    let dialect = declared.for_endpoint(ctx.base_url);
    let downgraded = declared == ChatDialect::OpenAi && dialect == ChatDialect::Compatible;

    let mut body = json!({
        "model": ctx.model,
        "messages": lower_messages(request, dialect, &mut warnings),
    });
    let object = body.as_object_mut().expect("body is an object");

    insert_tools(request, object, ToolShape::Nested);
    if let Some(output) = &request.structured_output {
        object.insert(
            "response_format".into(),
            json!({"type": "json_schema", "json_schema": json_schema_format(output)}),
        );
    }
    insert_reasoning(ctx, dialect, object, &mut warnings);
    insert_generation_settings(request, dialect, downgraded, object, &mut warnings);

    if streaming {
        object.insert("stream".into(), json!(true));
        let mut stream_options = json!({"include_usage": true});
        // Obfuscation padding is an OpenAI extension.
        if dialect != ChatDialect::Compatible {
            stream_options["include_obfuscation"] = json!(false);
        }
        object.insert("stream_options".into(), stream_options);
    }

    let namespace = dialect.profile().namespace();
    if let Some(options) = request.provider_options.get(namespace) {
        crate::util::json_merge(&mut body, options.clone());
    }

    let url = join_url(ctx.base_url, "chat/completions");
    Ok(LoweredRequest {
        http: HttpRequest::post_json(url, &body)?,
        warnings,
    })
}

fn insert_reasoning(
    ctx: &ProtocolContext<'_>,
    dialect: ChatDialect,
    object: &mut Map<String, Value>,
    warnings: &mut Vec<Warning>,
) {
    let Some(config) = ctx.request.reasoning else {
        return;
    };
    let Some((resolved, output)) =
        resolve_reasoning(config, &ctx.capabilities.reasoning, ctx.profile, warnings)
    else {
        return;
    };

    match dialect {
        ChatDialect::OpenAi | ChatDialect::Compatible | ChatDialect::Xai => {
            let effort = match resolved {
                ResolvedReasoning::Disabled => "none",
                ResolvedReasoning::Effort(effort) => effort.as_str(),
                ResolvedReasoning::Budget(_) => {
                    warnings.push(Warning::unsupported_setting(
                        "reasoning.budget",
                        format!("{} has no reasoning token budget", ctx.profile),
                    ));
                    return;
                }
            };
            object.insert("reasoning_effort".into(), json!(effort));
        }
        ChatDialect::OpenRouter => {
            let mut config = serde_json::Map::new();
            match resolved {
                ResolvedReasoning::Disabled => {
                    config.insert("effort".into(), json!("none"));
                }
                ResolvedReasoning::Effort(effort) => {
                    config.insert("effort".into(), json!(effort.as_str()));
                }
                ResolvedReasoning::Budget(tokens) => {
                    config.insert("max_tokens".into(), json!(tokens.get()));
                }
            }
            if let Some(output) = output {
                config.insert(
                    "exclude".into(),
                    json!(matches!(output, ReasoningOutput::Omit)),
                );
            }
            object.insert("reasoning".into(), Value::Object(config));
        }
    }
}

fn insert_generation_settings(
    request: &Request,
    dialect: ChatDialect,
    downgraded: bool,
    object: &mut Map<String, Value>,
    warnings: &mut Vec<Warning>,
) {
    if let Some(max) = request.max_output_tokens {
        // OpenAI deprecated `max_tokens` in favour of `max_completion_tokens`,
        // but other servers still spell it the original way.
        let key = match dialect {
            ChatDialect::OpenAi | ChatDialect::Xai => "max_completion_tokens",
            ChatDialect::Compatible | ChatDialect::OpenRouter => "max_tokens",
        };
        object.insert(key.into(), json!(max));
        // The downgrade changes the output-cap spelling, which OpenAI itself
        // rejects. Surface it so a gateway in front of OpenAI is debuggable.
        if downgraded {
            warnings.push(Warning::approximated_setting(
                "max_output_tokens",
                "base URL is not an OpenAI endpoint, so the generic Chat \
                 Completions dialect was used: `max_tokens` was sent instead \
                 of `max_completion_tokens`",
            ));
        }
    }
    if let Some(temperature) = request.temperature {
        object.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = request.top_p {
        object.insert("top_p".into(), json!(top_p));
    }
    if let Some(top_k) = request.top_k {
        object.insert("top_k".into(), json!(top_k));
    }
    if !request.stop_sequences.is_empty() {
        object.insert("stop".into(), json!(request.stop_sequences));
    }
    if let Some(penalty) = request.presence_penalty {
        object.insert("presence_penalty".into(), json!(penalty));
    }
    if let Some(penalty) = request.frequency_penalty {
        object.insert("frequency_penalty".into(), json!(penalty));
    }
    if let Some(seed) = request.seed {
        object.insert("seed".into(), json!(seed));
    }
}

pub(crate) fn lower_messages(
    request: &Request,
    dialect: ChatDialect,
    warnings: &mut Vec<Warning>,
) -> Value {
    let mut messages: Vec<Value> = Vec::new();
    if let Some(system) = request.system_prompt() {
        messages.push(json!({"role": "system", "content": system}));
    }
    for message in &request.messages {
        match message {
            Message::User { content } => messages.push(lower_user_message(content)),
            Message::Assistant {
                content,
                provider_metadata,
            } => {
                let origin = foreign_origin(provider_metadata, dialect.profile());
                if let Some(message) = lower_assistant_message(content, dialect, origin, warnings) {
                    messages.push(message);
                }
            }
            Message::Tool { content } => {
                for result in content {
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": result.content.to_text(),
                    }));
                }
            }
        }
    }
    Value::Array(messages)
}

fn lower_user_message(content: &[UserPart]) -> Value {
    // A single text part uses the plain-string form every server accepts.
    // Several parts stay separate content parts instead of being fused.
    let content = match content {
        [UserPart::Text { text }] => json!(text),
        parts => Value::Array(
            parts
                .iter()
                .map(|part| {
                    let UserPart::Text { text } = part;
                    json!({"type": "text", "text": text})
                })
                .collect(),
        ),
    };
    json!({"role": "user", "content": content})
}

fn lower_assistant_message(
    content: &[AssistantPart],
    dialect: ChatDialect,
    origin: Option<&str>,
    warnings: &mut Vec<Warning>,
) -> Option<Value> {
    let mut text = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut reasoning_details: Vec<Value> = Vec::new();
    let mut reasoning_text = String::new();
    let mut plaintext_reasoning = std::collections::BTreeMap::<&str, String>::new();
    for part in content {
        match part {
            AssistantPart::Text {
                text: part_text, ..
            } => text.push_str(part_text),
            AssistantPart::ToolCall(call) => tool_calls.push(json!({
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments,
                },
            })),
            AssistantPart::Reasoning(reasoning) => {
                if dialect == ChatDialect::Compatible && origin.is_none() {
                    let field = reasoning
                        .provider_metadata
                        .get("openai")
                        .and_then(|metadata| metadata.get("reasoning_field"))
                        .and_then(Value::as_str);
                    if let Some(field @ ("reasoning_content" | "reasoning")) = field {
                        plaintext_reasoning
                            .entry(field)
                            .or_default()
                            .push_str(&reasoning.visible_text());
                    }
                }
                if dialect != ChatDialect::OpenRouter {
                    continue;
                }
                // Reasoning from another profile is not rebuilt into
                // `reasoning_details`: its encrypted payloads cannot be
                // verified by this backend.
                if let Some(origin) = origin {
                    if reasoning
                        .content
                        .iter()
                        .any(|content| matches!(content, ReasoningContent::Encrypted { .. }))
                    {
                        warnings.push(Warning::other(format!(
                            "dropped opaque reasoning items produced by `{origin}`; `{}` cannot \
                             verify state from another provider",
                            dialect.profile().as_str()
                        )));
                    }
                    continue;
                }
                collect_openrouter_reasoning(
                    reasoning,
                    &mut reasoning_details,
                    &mut reasoning_text,
                );
            }
            AssistantPart::Compaction(compaction) => {
                if let Some(summary) = &compaction.content {
                    text.push_str(summary);
                    warnings.push(Warning::other(
                        "compaction summary was flattened to plain assistant text: this protocol \
                         has no native compaction item",
                    ));
                }
            }
            AssistantPart::ProviderTool { .. } => warnings.push(Warning::other(
                "server-tool item dropped: this protocol cannot replay provider-executed tool \
                 items",
            )),
        }
    }

    if text.is_empty()
        && tool_calls.is_empty()
        && reasoning_details.is_empty()
        && reasoning_text.is_empty()
        && plaintext_reasoning.is_empty()
    {
        return None;
    }
    let mut assistant = Map::new();
    assistant.insert("role".into(), json!("assistant"));
    assistant.insert(
        "content".into(),
        if text.is_empty() { Value::Null } else { json!(text) },
    );
    if !tool_calls.is_empty() {
        assistant.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    if !reasoning_details.is_empty() {
        assistant.insert("reasoning_details".into(), Value::Array(reasoning_details));
    } else if !reasoning_text.is_empty() {
        assistant.insert("reasoning".into(), json!(reasoning_text));
    }
    for (field, text) in plaintext_reasoning {
        assistant.insert(field.into(), json!(text));
    }
    Some(Value::Object(assistant))
}

/// Rebuild OpenRouter `reasoning_details` for replay: verbatim provider
/// metadata when present, otherwise reconstructed from typed content.
pub(crate) fn collect_openrouter_reasoning(
    reasoning: &ReasoningPart,
    details_out: &mut Vec<Value>,
    text_out: &mut String,
) {
    if let Some(Value::Object(namespace)) = reasoning.provider_metadata.get("openrouter")
        && let Some(Value::Array(details)) = namespace.get("reasoning_details")
    {
        details_out.extend(details.iter().cloned());
        return;
    }
    for content in &reasoning.content {
        match content {
            ReasoningContent::Text { text, signature } => {
                let mut detail = json!({"type": "reasoning.text", "text": text});
                if let Some(signature) = signature {
                    detail["signature"] = json!(signature);
                }
                details_out.push(detail);
                text_out.push_str(text);
            }
            ReasoningContent::Summary { text } => {
                details_out.push(json!({"type": "reasoning.summary", "summary": text}));
                text_out.push_str(text);
            }
            ReasoningContent::Encrypted { data } | ReasoningContent::Redacted { data } => {
                details_out.push(json!({"type": "reasoning.encrypted", "data": data}));
            }
        }
    }
}
