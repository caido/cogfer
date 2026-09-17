use std::num::NonZeroU32;

use serde_json::{Value, json};

use super::AnthropicDialect;
use crate::error::{Error, ErrorKind, Result};
#[cfg(feature = "aws")]
use crate::http::encode_path_segment;
use crate::http::join_url;
use crate::message::{AssistantPart, Message, ReasoningContent, ReasoningPart, ToolCall, UserPart};
use crate::protocols::{LoweredRequest, ProtocolContext, ResolvedReasoning, resolve_reasoning};
use crate::request::{ReasoningConfig, ReasoningOutput, Request, ToolChoice};
use crate::response::Warning;
#[cfg(feature = "aws")]
use crate::transport::header;
use crate::transport::{HeaderName, HeaderValue, HttpRequest};

/// Anthropic's `max_tokens` is mandatory. This is the value used when the
/// caller leaves the output cap to the SDK.
const FALLBACK_MAX_TOKENS: u32 = 8192;

/// Replayed tool-call arguments as the `tool_use` input object. Request
/// validation already guarantees they parse as a JSON object.
fn tool_input(call: &ToolCall) -> Result<Value> {
    call.arguments_value().map_err(|source| {
        Error::invalid_request(format!(
            "anthropic tool call `{}` has invalid JSON arguments: {source}",
            call.call_id
        ))
        .with_source(source)
    })
}

/// Anthropic rejects empty text blocks and messages without content, which
/// can appear in histories accumulated from other providers.
fn text_block(text: &str) -> Option<Value> {
    (!text.is_empty()).then(|| json!({"type": "text", "text": text}))
}

pub(crate) fn lower_messages(request: &crate::request::Request) -> Result<Value> {
    let mut messages: Vec<Value> = Vec::new();
    for message in &request.messages {
        match message {
            Message::User { content } => {
                let blocks: Vec<Value> = content
                    .iter()
                    .filter_map(|part| {
                        let UserPart::Text { text } = part;
                        text_block(text)
                    })
                    .collect();
                if !blocks.is_empty() {
                    messages.push(json!({"role": "user", "content": blocks}));
                }
            }
            Message::Assistant { content, .. } => {
                let mut blocks = Vec::new();
                for part in content {
                    match part {
                        AssistantPart::Text { text, .. } => {
                            blocks.extend(text_block(text));
                        }
                        AssistantPart::Reasoning(reasoning) => {
                            lower_reasoning(reasoning, &mut blocks);
                        }
                        AssistantPart::ToolCall(call) => {
                            blocks.push(json!({
                                "type": "tool_use",
                                "id": call.call_id,
                                "name": call.name,
                                "input": tool_input(call)?,
                            }));
                        }
                        AssistantPart::Compaction(compaction) => {
                            if let Some(text) = &compaction.content {
                                blocks.push(json!({"type": "compaction", "content": text}));
                            } else {
                                // Without summary text there is nothing to replay: the item
                                // is either another provider's opaque state or a compaction
                                // that produced no content.
                                return Err(Error::new(
                                    ErrorKind::UnsupportedContent,
                                    "history contains a compaction item without summary \
                                     content; anthropic can only replay summary compactions",
                                ));
                            }
                        }
                        AssistantPart::ProviderTool {
                            provider_tool: tool,
                        } => {
                            // Foreign server tool items have no Anthropic representation.
                            if tool.namespace == "anthropic" {
                                blocks.push(tool.payload.clone());
                            }
                        }
                    }
                }
                if !blocks.is_empty() {
                    messages.push(json!({"role": "assistant", "content": blocks}));
                }
            }
            Message::Tool { content } => {
                let mut blocks = Vec::new();
                for result in content {
                    let mut block = json!({
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": result.content.to_text(),
                    });
                    if result.is_error {
                        block["is_error"] = json!(true);
                    }
                    blocks.push(block);
                }
                // Anthropic requires one user message for all results from a tool turn.
                if let Some(previous) = messages.last_mut()
                    && previous["role"] == json!("user")
                    && previous["content"].as_array().is_some_and(|existing| {
                        existing
                            .iter()
                            .all(|block| block["type"] == json!("tool_result"))
                    })
                    && let Some(existing) = previous["content"].as_array_mut()
                {
                    existing.extend(blocks);
                    continue;
                }
                messages.push(json!({"role": "user", "content": blocks}));
            }
        }
    }
    Ok(Value::Array(messages))
}

/// Replay Anthropic thinking verbatim. Unsigned or foreign reasoning content
/// is skipped because Anthropic rejects it on replay.
pub(crate) fn lower_reasoning(reasoning: &ReasoningPart, blocks: &mut Vec<Value>) {
    for content in &reasoning.content {
        match content {
            ReasoningContent::Text {
                text,
                signature: Some(signature),
            } => {
                blocks.push(json!({
                    "type": "thinking",
                    "thinking": text,
                    "signature": signature,
                }));
            }
            ReasoningContent::Redacted { data } => {
                blocks.push(json!({"type": "redacted_thinking", "data": data}));
            }
            ReasoningContent::Text { .. }
            | ReasoningContent::Summary { .. }
            | ReasoningContent::Encrypted { .. } => {}
        }
    }
}

/// The `max_tokens` to send: the caller's cap, or the fallback plus any
/// thinking budget so the budget never starves the visible answer.
fn effective_max_tokens(request: &Request, reasoning: Option<ResolvedReasoning>) -> u32 {
    if let Some(max) = request.max_output_tokens {
        return max;
    }
    match reasoning {
        Some(ResolvedReasoning::Budget(tokens)) => FALLBACK_MAX_TOKENS.saturating_add(tokens.get()),
        _ => FALLBACK_MAX_TOKENS,
    }
}

/// A budget derived from a requested effort must still fit under the caller's
/// output cap (Anthropic requires `budget_tokens < max_tokens`). The caller
/// never chose the number, so shrink it, or drop reasoning when even the
/// 1024-token minimum cannot fit, instead of failing the request. Explicit
/// budgets keep their hard error in [`lower_request_reasoning`].
fn fit_derived_budget_to_output_cap(
    request: &Request,
    reasoning: Option<(ResolvedReasoning, Option<ReasoningOutput>)>,
    warnings: &mut Vec<Warning>,
) -> Option<(ResolvedReasoning, Option<ReasoningOutput>)> {
    let (ResolvedReasoning::Budget(tokens), output) = reasoning? else {
        return reasoning;
    };
    if !matches!(request.reasoning, Some(ReasoningConfig::Effort { .. })) {
        return reasoning;
    }
    let Some(max) = request.max_output_tokens else {
        return reasoning;
    };
    if tokens.get() < max {
        return reasoning;
    }
    let available = max.saturating_sub(1);
    if let Some(available) = NonZeroU32::new(available).filter(|a| a.get() >= 1024) {
        warnings.push(Warning::approximated_setting(
            "reasoning.effort",
            format!(
                "thinking budget reduced to {available} tokens to stay below \
                 max_output_tokens {max}",
            ),
        ));
        return Some((ResolvedReasoning::Budget(available), output));
    }
    warnings.push(Warning::unsupported_setting(
        "reasoning.effort",
        format!(
            "the approximated thinking budget was dropped: max_output_tokens \
             {max} cannot fit the 1024-token minimum",
        ),
    ));
    None
}

fn lower_request_reasoning(
    request: &Request,
    reasoning: Option<(ResolvedReasoning, Option<ReasoningOutput>)>,
    max_tokens: u32,
    object: &mut serde_json::Map<String, Value>,
) -> Result<()> {
    let mut output_config = serde_json::Map::new();
    if let Some((resolved, output)) = reasoning {
        let display = output.map(|output| match output {
            ReasoningOutput::Include => "summarized",
            ReasoningOutput::Omit => "omitted",
        });
        match resolved {
            ResolvedReasoning::Disabled => {
                object.insert("thinking".into(), json!({"type": "disabled"}));
            }
            ResolvedReasoning::Effort(effort) => {
                let mut thinking = json!({"type": "adaptive"});
                if let Some(display) = display {
                    thinking["display"] = json!(display);
                }
                object.insert("thinking".into(), thinking);
                output_config.insert("effort".into(), json!(effort.as_str()));
            }
            ResolvedReasoning::Budget(tokens) => {
                let tokens = tokens.get();
                if tokens < 1024 {
                    return Err(Error::invalid_request(
                        "anthropic reasoning budgets must be at least 1024 tokens",
                    ));
                }
                if request.max_output_tokens.is_some_and(|max| tokens >= max) {
                    return Err(Error::invalid_request(format!(
                        "anthropic reasoning budget {tokens} must be less than max_output_tokens {max_tokens}",
                    )));
                }
                let mut thinking = json!({
                    "type": "enabled",
                    "budget_tokens": tokens,
                });
                if let Some(display) = display {
                    thinking["display"] = json!(display);
                }
                object.insert("thinking".into(), thinking);
            }
        }
    }
    if let Some(output) = &request.structured_output {
        output_config.insert(
            "format".into(),
            json!({"type": "json_schema", "schema": output.schema}),
        );
    }
    if !output_config.is_empty() {
        object.insert("output_config".into(), Value::Object(output_config));
    }
    Ok(())
}

/// Anthropic rejects compaction triggers below this many input tokens.
const MIN_COMPACTION_TRIGGER: u64 = 50_000;

fn lower_compaction(
    request: &Request,
    object: &mut serde_json::Map<String, Value>,
    warnings: &mut Vec<Warning>,
) -> Vec<&'static str> {
    let history_has_compaction = request.messages.iter().any(|message| {
        matches!(message, Message::Assistant { content, .. }
        if content.iter().any(|part| matches!(
            part,
            AssistantPart::Compaction(compaction) if compaction.content.is_some()
        )))
    });
    if request.compaction.is_none() && !history_has_compaction {
        return Vec::new();
    }

    let mut edit = serde_json::Map::new();
    edit.insert("type".into(), json!("compact_20260112"));
    if let Some(compaction) = &request.compaction {
        if let Some(trigger) = compaction.trigger_input_tokens {
            let clamped = trigger.max(MIN_COMPACTION_TRIGGER);
            if clamped != trigger {
                warnings.push(Warning::other(format!(
                    "compaction trigger raised to the API minimum \
                     {MIN_COMPACTION_TRIGGER} (requested {trigger})",
                )));
            }
            edit.insert(
                "trigger".into(),
                json!({"type": "input_tokens", "value": clamped}),
            );
        }
        if let Some(pause) = compaction.pause_after_compaction {
            edit.insert("pause_after_compaction".into(), json!(pause));
        }
        if let Some(instructions) = &compaction.instructions {
            edit.insert("instructions".into(), json!(instructions));
        }
    }
    object.insert(
        "context_management".into(),
        json!({"edits": [Value::Object(edit)]}),
    );
    vec!["compact-2026-01-12"]
}

fn lower_tools(request: &Request, object: &mut serde_json::Map<String, Value>) {
    if request.tools.is_empty() {
        return;
    }
    let tools = request
        .tools
        .iter()
        .map(|tool| {
            let mut definition = json!({
                "name": tool.name,
                "input_schema": tool.parameters,
            });
            if let Some(description) = &tool.description {
                definition["description"] = json!(description);
            }
            if tool.strict == Some(true) {
                definition["strict"] = json!(true);
            }
            definition
        })
        .collect();
    object.insert("tools".into(), Value::Array(tools));
}

fn lower_tool_choice(
    request: &Request,
    budget_thinking: bool,
    object: &mut serde_json::Map<String, Value>,
) -> Result<()> {
    let disable_parallel = request.parallel_tool_calls == Some(false);
    let tool_choice = match &request.tool_choice {
        Some(ToolChoice::Auto) => Some(json!({"type": "auto"})),
        Some(ToolChoice::None) => Some(json!({"type": "none"})),
        Some(ToolChoice::Required) => Some(json!({"type": "any"})),
        Some(ToolChoice::Tool { name }) => Some(json!({"type": "tool", "name": name})),
        None if disable_parallel => Some(json!({"type": "auto"})),
        None => None,
    };
    let Some(mut choice) = tool_choice else {
        return Ok(());
    };
    if budget_thinking && matches!(choice["type"].as_str(), Some("any" | "tool")) {
        return Err(Error::invalid_request(
            "anthropic manual reasoning budgets cannot be combined with forced tool choice",
        ));
    }
    if disable_parallel && choice["type"] != json!("none") {
        choice["disable_parallel_tool_use"] = json!(true);
    }
    object.insert("tool_choice".into(), choice);
    Ok(())
}

fn lower_sampling(
    request: &Request,
    thinking_enabled: bool,
    object: &mut serde_json::Map<String, Value>,
    warnings: &mut Vec<Warning>,
) {
    // Some models allow a narrow sampling range during thinking. We omit all
    // sampling settings to keep the behavior consistent across models.
    for (setting, present) in [
        ("temperature", request.temperature.is_some()),
        ("top_p", request.top_p.is_some()),
        ("top_k", request.top_k.is_some()),
    ] {
        if thinking_enabled && present {
            warnings.push(Warning::unsupported_setting(
                setting,
                format!("cogfer omits `{setting}` when thinking is enabled"),
            ));
        }
    }
    if !thinking_enabled {
        if let Some(temperature) = request.temperature {
            object.insert("temperature".into(), json!(temperature));
        }
        if let Some(top_p) = request.top_p {
            object.insert("top_p".into(), json!(top_p));
        }
        if let Some(top_k) = request.top_k {
            object.insert("top_k".into(), json!(top_k));
        }
    }
}

pub(crate) fn lower_anthropic_request(
    ctx: &ProtocolContext<'_>,
    streaming: bool,
    dialect: AnthropicDialect,
) -> Result<LoweredRequest> {
    let mut warnings = Vec::new();
    let request = ctx.request;
    let reasoning = request.reasoning.and_then(|config| {
        resolve_reasoning(
            config,
            &ctx.capabilities.reasoning,
            ctx.profile,
            &mut warnings,
        )
    });
    let reasoning = fit_derived_budget_to_output_cap(request, reasoning, &mut warnings);
    let max_tokens = effective_max_tokens(request, reasoning.map(|(resolved, _)| resolved));

    let mut body = json!({
        "max_tokens": max_tokens,
        "messages": lower_messages(request)?,
    });
    let object = body.as_object_mut().expect("body is an object");
    match dialect {
        AnthropicDialect::Direct => {
            object.insert("model".into(), json!(ctx.model));
        }
        // Bedrock takes the model from the URL and the API version from the body.
        #[cfg(feature = "aws")]
        AnthropicDialect::Bedrock => {
            object.insert(
                "anthropic_version".into(),
                json!(super::bedrock::ANTHROPIC_VERSION),
            );
        }
    }

    if let Some(system) = request.system_prompt() {
        object.insert("system".into(), json!(system));
    }
    lower_tools(request, object);
    let budget_thinking = matches!(reasoning, Some((ResolvedReasoning::Budget(_), _)));
    // Manual budgets need separate forced-tool-choice validation. Sampling
    // settings are omitted in both thinking modes.
    let thinking_enabled =
        budget_thinking || matches!(reasoning, Some((ResolvedReasoning::Effort(_), _)));
    lower_request_reasoning(request, reasoning, max_tokens, object)?;
    lower_tool_choice(request, budget_thinking, object)?;
    let beta_features = lower_compaction(request, object, &mut warnings);
    // Bedrock takes beta opt-ins in the body; the direct API takes them in the
    // `anthropic-beta` header below.
    #[cfg(feature = "aws")]
    if dialect == AnthropicDialect::Bedrock && !beta_features.is_empty() {
        object.insert("anthropic_beta".into(), json!(beta_features));
    }
    lower_sampling(request, thinking_enabled, object, &mut warnings);
    if !request.stop_sequences.is_empty() {
        object.insert("stop_sequences".into(), json!(request.stop_sequences));
    }
    // Bedrock selects streaming by endpoint rather than a body flag.
    if streaming && dialect == AnthropicDialect::Direct {
        object.insert("stream".into(), json!(true));
    }
    if let Some(options) = request.provider_options.get("anthropic") {
        crate::util::json_merge(&mut body, options.clone());
    }

    let http = match dialect {
        AnthropicDialect::Direct => {
            let mut http = HttpRequest::post_json(join_url(ctx.base_url, "messages"), &body)?;
            http.headers.insert(super::VERSION_HEADER, super::VERSION);
            if !beta_features.is_empty() {
                http.headers.insert(
                    HeaderName::from_static("anthropic-beta"),
                    HeaderValue::from_str(&beta_features.join(","))
                        .expect("beta feature names are valid header values"),
                );
            }
            http
        }
        #[cfg(feature = "aws")]
        AnthropicDialect::Bedrock => {
            let operation = if streaming { "invoke-with-response-stream" } else { "invoke" };
            let path = format!("model/{}/{operation}", encode_path_segment(ctx.model));
            let mut http = HttpRequest::post_json(join_url(ctx.base_url, &path), &body)?;
            http.headers
                .insert(header::ACCEPT, HeaderValue::from_static("application/json"));
            http
        }
    };
    Ok(LoweredRequest { http, warnings })
}
