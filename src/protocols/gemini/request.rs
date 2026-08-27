use std::collections::HashMap;

use serde_json::{Map, Value, json};

use super::SIGNATURE_KEY;
use crate::error::{Error, Result};
use crate::http::join_url;
use crate::message::{AssistantPart, Message, ToolResultContent, ToolResultPart, UserPart};
use crate::metadata::ProviderMetadata;
use crate::protocols::{LoweredRequest, ProtocolContext, ResolvedReasoning, resolve_reasoning};
use crate::request::{ReasoningOutput, Request, ToolChoice};
use crate::response::Warning;
use crate::transport::HttpRequest;

pub(crate) fn signature_of(metadata: &ProviderMetadata) -> Option<String> {
    metadata
        .get("gemini")
        .and_then(|namespace| namespace.get(SIGNATURE_KEY))
        .and_then(Value::as_str)
        .map(str::to_string)
}

type CallIndex = HashMap<String, (String, Option<String>)>;

fn lower_user_parts(content: &[UserPart]) -> Vec<Value> {
    content
        .iter()
        .map(|part| {
            let UserPart::Text { text } = part;
            json!({"text": text})
        })
        .collect()
}

fn lower_assistant_parts(
    content: &[AssistantPart],
    call_index: &mut CallIndex,
    warnings: &mut Vec<Warning>,
) -> Result<Vec<Value>> {
    let mut parts = Vec::new();
    for part in content {
        match part {
            AssistantPart::Text {
                text,
                provider_metadata,
            } => {
                let mut value = json!({"text": text});
                if let Some(signature) = signature_of(provider_metadata) {
                    value["thoughtSignature"] = json!(signature);
                }
                parts.push(value);
            }
            AssistantPart::ToolCall(call) => {
                // Request validation already guarantees a JSON object.
                let args = call.arguments_value().map_err(|source| {
                    Error::invalid_request(format!(
                        "gemini tool call `{}` has invalid JSON arguments: {source}",
                        call.call_id
                    ))
                    .with_source(source)
                })?;
                call_index.insert(
                    call.call_id.clone(),
                    (call.name.clone(), call.item_id.clone()),
                );
                let mut function_call = json!({"name": call.name, "args": args});
                if let Some(id) = &call.item_id {
                    function_call["id"] = json!(id);
                }
                let mut value = json!({"functionCall": function_call});
                if let Some(signature) = signature_of(&call.provider_metadata) {
                    value["thoughtSignature"] = json!(signature);
                }
                parts.push(value);
            }
            AssistantPart::Reasoning(reasoning) => {
                if let Some(signature) = signature_of(&reasoning.provider_metadata) {
                    parts.push(json!({
                        "text": reasoning.visible_text(),
                        "thought": true,
                        "thoughtSignature": signature,
                    }));
                }
            }
            AssistantPart::Compaction(compaction) => {
                if let Some(summary) = &compaction.content {
                    parts.push(json!({"text": summary}));
                    warnings.push(Warning::other(
                        "compaction summary was flattened to plain assistant text: \
                         this protocol has no native compaction item",
                    ));
                }
            }
            AssistantPart::ProviderTool { .. } => {
                warnings.push(Warning::other(
                    "server-tool item dropped: this protocol cannot replay \
                     provider-executed tool items",
                ));
            }
        }
    }
    Ok(parts)
}

fn lower_tool_parts(content: &[ToolResultPart], call_index: &CallIndex) -> Result<Vec<Value>> {
    content
        .iter()
        .map(|result| {
            let (name, provider_id) =
                call_index.get(&result.call_id).cloned().unwrap_or_else(|| {
                    (
                        result.name.clone().unwrap_or_default(),
                        result.item_id.clone(),
                    )
                });
            let name = result.name.clone().unwrap_or(name);
            if name.is_empty() {
                return Err(Error::invalid_request(format!(
                    "gemini requires the function name on tool results; none found for call `{}`",
                    result.call_id
                )));
            }
            let response = match (&result.content, result.is_error) {
                (ToolResultContent::Json { value }, true) => json!({"error": value}),
                (other, true) => json!({"error": other.to_text()}),
                (
                    ToolResultContent::Json {
                        value: Value::Object(map),
                    },
                    false,
                ) => Value::Object(map.clone()),
                (other, false) => json!({"result": other.to_text()}),
            };
            let mut function_response = json!({"name": name, "response": response});
            if let Some(id) = provider_id.or_else(|| result.item_id.clone()) {
                function_response["id"] = json!(id);
            }
            Ok(json!({"functionResponse": function_response}))
        })
        .collect()
}

fn append_tool_content(contents: &mut Vec<Value>, parts: Vec<Value>) {
    // Gemini requires one user turn for all function responses.
    if let Some(previous) = contents.last_mut() {
        let is_function_response_turn = previous["role"] == json!("user")
            && previous["parts"].as_array().is_some_and(|existing_parts| {
                existing_parts
                    .iter()
                    .all(|part| part.get("functionResponse").is_some())
            });
        if is_function_response_turn && let Some(existing_parts) = previous["parts"].as_array_mut()
        {
            existing_parts.extend(parts);
            return;
        }
    }
    contents.push(json!({"role": "user", "parts": parts}));
}

pub(crate) fn lower_contents(
    request: &crate::request::Request,
    warnings: &mut Vec<Warning>,
) -> Result<Value> {
    let mut contents = Vec::new();
    let mut call_index = CallIndex::new();

    for message in &request.messages {
        match message {
            Message::User { content } => {
                let parts = lower_user_parts(content);
                contents.push(json!({"role": "user", "parts": parts}));
            }
            Message::Assistant { content, .. } => {
                let parts = lower_assistant_parts(content, &mut call_index, warnings)?;
                if !parts.is_empty() {
                    contents.push(json!({"role": "model", "parts": parts}));
                }
            }
            Message::Tool { content } => {
                let parts = lower_tool_parts(content, &call_index)?;
                append_tool_content(&mut contents, parts);
            }
        }
    }
    Ok(Value::Array(contents))
}

fn add_system_instruction(object: &mut Map<String, Value>, request: &Request) {
    if let Some(system) = request.system_prompt() {
        object.insert(
            "systemInstruction".into(),
            json!({"parts": [{"text": system}]}),
        );
    }
}

fn add_tool_declarations(object: &mut Map<String, Value>, request: &Request) {
    if request.tools.is_empty() {
        return;
    }
    let declarations: Vec<Value> = request
        .tools
        .iter()
        .map(|tool| {
            let mut declaration = json!({
                "name": tool.name,
                "parametersJsonSchema": tool.parameters,
            });
            if let Some(description) = &tool.description {
                declaration["description"] = json!(description);
            }
            declaration
        })
        .collect();
    object.insert(
        "tools".into(),
        json!([{"functionDeclarations": declarations}]),
    );
}

fn add_tool_config(object: &mut Map<String, Value>, request: &Request) {
    if let Some(choice) = &request.tool_choice {
        let config = match choice {
            ToolChoice::Auto => json!({"mode": "AUTO"}),
            ToolChoice::None => json!({"mode": "NONE"}),
            ToolChoice::Required => json!({"mode": "ANY"}),
            ToolChoice::Tool { name } => {
                json!({"mode": "ANY", "allowedFunctionNames": [name]})
            }
        };
        object.insert(
            "toolConfig".into(),
            json!({"functionCallingConfig": config}),
        );
    }
}

fn thinking_config(ctx: &ProtocolContext<'_>, warnings: &mut Vec<Warning>) -> Option<Value> {
    let config = ctx.request.reasoning?;
    let (resolved, output) =
        resolve_reasoning(config, &ctx.capabilities.reasoning, ctx.profile, warnings)?;
    let mut thinking = Map::new();
    match resolved {
        ResolvedReasoning::Disabled => {
            thinking.insert("thinkingBudget".into(), json!(0));
        }
        ResolvedReasoning::Effort(effort) => {
            thinking.insert("thinkingLevel".into(), json!(effort.as_str()));
        }
        ResolvedReasoning::Budget(tokens) => {
            thinking.insert("thinkingBudget".into(), json!(tokens.get()));
        }
    }
    if let Some(output) = output {
        thinking.insert(
            "includeThoughts".into(),
            json!(matches!(output, ReasoningOutput::Include)),
        );
    }
    Some(Value::Object(thinking))
}

fn generation_config(ctx: &ProtocolContext<'_>, warnings: &mut Vec<Warning>) -> Map<String, Value> {
    let request = ctx.request;
    let mut generation = Map::new();
    if let Some(temperature) = request.temperature {
        generation.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = request.top_p {
        generation.insert("topP".into(), json!(top_p));
    }
    if let Some(top_k) = request.top_k {
        generation.insert("topK".into(), json!(top_k));
    }
    if let Some(max) = request.max_output_tokens {
        generation.insert("maxOutputTokens".into(), json!(max));
    }
    if !request.stop_sequences.is_empty() {
        generation.insert("stopSequences".into(), json!(request.stop_sequences));
    }
    if let Some(seed) = request.seed {
        generation.insert("seed".into(), json!(seed));
    }
    if let Some(penalty) = request.presence_penalty {
        generation.insert("presencePenalty".into(), json!(penalty));
    }
    if let Some(penalty) = request.frequency_penalty {
        generation.insert("frequencyPenalty".into(), json!(penalty));
    }
    if let Some(output) = &request.structured_output {
        generation.insert("responseMimeType".into(), json!("application/json"));
        generation.insert("responseJsonSchema".into(), output.schema.clone());
    }
    if let Some(thinking) = thinking_config(ctx, warnings) {
        generation.insert("thinkingConfig".into(), thinking);
    }
    generation
}

fn generate_url(ctx: &ProtocolContext<'_>, streaming: bool) -> url::Url {
    let model = ctx.model.strip_prefix("models/").unwrap_or(ctx.model);
    let method = if streaming { "streamGenerateContent" } else { "generateContent" };
    let mut url = join_url(ctx.base_url, &format!("models/{model}:{method}"));
    if streaming {
        url.query_pairs_mut().append_pair("alt", "sse");
    }
    url
}

pub(crate) fn lower_gemini_request(
    ctx: &ProtocolContext<'_>,
    streaming: bool,
) -> Result<LoweredRequest> {
    let mut warnings = Vec::new();
    let request = ctx.request;
    let mut body = json!({
        "contents": lower_contents(request, &mut warnings)?,
    });
    let object = body.as_object_mut().expect("body is an object");
    add_system_instruction(object, request);
    add_tool_declarations(object, request);
    add_tool_config(object, request);
    let generation = generation_config(ctx, &mut warnings);
    if !generation.is_empty() {
        object.insert("generationConfig".into(), Value::Object(generation));
    }

    if let Some(options) = request.provider_options.get("gemini") {
        crate::util::json_merge(&mut body, options.clone());
    }

    Ok(LoweredRequest {
        http: HttpRequest::post_json(generate_url(ctx, streaming), &body)?,
        warnings,
    })
}
