//! Pre-flight validation for constraints enforced by this crate.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::capabilities::ModelCapabilities;
use crate::error::{Error, ErrorKind, Result};
use crate::message::{AssistantPart, Message, ToolCall, ToolResultPart};
use crate::protocols::ApiProfile;
use crate::request::{Request, ToolChoice};

pub(crate) fn validate(
    request: &Request,
    model: &str,
    profile: ApiProfile,
    capabilities: &ModelCapabilities,
) -> Result<()> {
    validate_structure(request, model, profile)
        .and_then(|()| validate_profile(request, profile, capabilities))
        .map_err(|error| error.with_origin(profile.as_str()).with_model(model))
}

fn validate_structure(request: &Request, model: &str, profile: ApiProfile) -> Result<()> {
    if model.trim().is_empty() {
        return Err(Error::invalid_request("model identifier must not be blank"));
    }
    if request.messages.is_empty() {
        return Err(Error::invalid_request("request has no messages"));
    }

    validate_message_shapes(request)?;
    validate_numbers(request)?;
    validate_tools(request, profile)?;
    validate_structured_output(request)?;
    validate_provider_options(request)?;
    validate_history(request)
}

/// Provider options are deep-merged into the wire body, so anything but an
/// object would replace the body wholesale.
fn validate_provider_options(request: &Request) -> Result<()> {
    for (namespace, options) in request.provider_options.iter() {
        if !options.is_object() {
            return Err(Error::invalid_request(format!(
                "provider_options.{namespace} must be a JSON object"
            )));
        }
    }
    Ok(())
}

fn validate_message_shapes(request: &Request) -> Result<()> {
    for (index, message) in request.messages.iter().enumerate() {
        if matches!(message, Message::Tool { content } if content.is_empty()) {
            return Err(Error::invalid_request(format!(
                "messages[{index}].content must contain at least one tool result"
            )));
        }
    }
    Ok(())
}

fn validate_numbers(request: &Request) -> Result<()> {
    for (name, value) in [
        ("temperature", request.temperature),
        ("top_p", request.top_p),
        ("presence_penalty", request.presence_penalty),
        ("frequency_penalty", request.frequency_penalty),
    ] {
        if value.is_some_and(|value| !value.is_finite()) {
            return Err(Error::invalid_request(format!(
                "{name} must be a finite number"
            )));
        }
    }
    if request.max_output_tokens == Some(0) {
        return Err(Error::invalid_request(
            "max_output_tokens must be greater than zero",
        ));
    }
    if request.top_k == Some(0) {
        return Err(Error::invalid_request("top_k must be greater than zero"));
    }
    Ok(())
}

fn validate_tools(request: &Request, profile: ApiProfile) -> Result<()> {
    let mut names = HashSet::with_capacity(request.tools.len());
    for (index, tool) in request.tools.iter().enumerate() {
        if tool.name.trim().is_empty() {
            return Err(Error::invalid_request(format!(
                "tools[{index}].name must not be blank"
            )));
        }
        if !is_json_schema(&tool.parameters) {
            return Err(Error::invalid_request(format!(
                "tools[{index}].parameters must be a JSON Schema object or boolean"
            )));
        }
        if !names.insert(tool.name.as_str()) {
            return Err(Error::invalid_request(format!(
                "tool name `{}` is declared more than once",
                tool.name
            )));
        }
    }

    if matches!(&request.tool_choice, Some(ToolChoice::Tool { name }) if name.trim().is_empty()) {
        return Err(Error::invalid_request(
            "tool_choice tool name must not be blank",
        ));
    }
    if !typed_tool_configuration_is_authoritative(request, profile) {
        return Ok(());
    }

    match &request.tool_choice {
        Some(ToolChoice::Required) if names.is_empty() => Err(Error::invalid_request(
            "tool_choice `required` requires at least one declared tool",
        )),
        Some(ToolChoice::Tool { name }) if !names.contains(name.as_str()) => Err(
            Error::invalid_request(format!("tool_choice references undeclared tool `{name}`")),
        ),
        _ => Ok(()),
    }
}

fn typed_tool_configuration_is_authoritative(request: &Request, profile: ApiProfile) -> bool {
    let Some(options) = request
        .provider_options
        .get(profile.namespace())
        .and_then(Value::as_object)
    else {
        return true;
    };
    let choice_key = if profile == ApiProfile::GeminiGenerateContent {
        "toolConfig"
    } else {
        "tool_choice"
    };
    !options.contains_key("tools") && !options.contains_key(choice_key)
}

fn is_json_schema(value: &Value) -> bool {
    value.is_object() || value.is_boolean()
}

fn validate_structured_output(request: &Request) -> Result<()> {
    if request
        .structured_output
        .as_ref()
        .is_some_and(|output| !is_json_schema(&output.schema))
    {
        return Err(Error::invalid_request(
            "structured_output.schema must be a JSON Schema object or boolean",
        ));
    }
    Ok(())
}

fn validate_history(request: &Request) -> Result<()> {
    let mut unresolved = HashMap::new();
    let mut resolved = HashSet::new();

    for (message_index, message) in request.messages.iter().enumerate() {
        match message {
            Message::Assistant { content, .. } => {
                for (part_index, part) in content.iter().enumerate() {
                    let AssistantPart::ToolCall(call) = part else {
                        continue;
                    };
                    let path = format!("messages[{message_index}].content[{part_index}]");
                    validate_tool_call(call, &path)?;
                    if unresolved.contains_key(call.call_id.as_str()) {
                        return Err(Error::invalid_request(format!(
                            "{path}.call_id `{}` duplicates an unresolved tool call",
                            call.call_id
                        )));
                    }
                    resolved.remove(call.call_id.as_str());
                    unresolved.insert(call.call_id.as_str(), call);
                }
            }
            Message::Tool { content } => {
                for (part_index, result) in content.iter().enumerate() {
                    let path = format!("messages[{message_index}].content[{part_index}]");
                    validate_tool_result(result, &path)?;
                    if let Some(call) = unresolved.remove(result.call_id.as_str()) {
                        validate_matching_identity(result, call, &path)?;
                        resolved.insert(result.call_id.as_str());
                    } else if resolved.contains(result.call_id.as_str()) {
                        return Err(Error::invalid_request(format!(
                            "{path}.call_id `{}` has more than one result",
                            result.call_id
                        )));
                    }
                }
            }
            Message::User { .. } => {}
        }
    }
    Ok(())
}

fn validate_tool_call(call: &ToolCall, path: &str) -> Result<()> {
    if call.call_id.trim().is_empty() {
        return Err(Error::invalid_request(format!(
            "{path}.call_id must not be blank"
        )));
    }
    if call.name.trim().is_empty() {
        return Err(Error::invalid_request(format!(
            "{path}.name must not be blank"
        )));
    }
    if call
        .item_id
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(Error::invalid_request(format!(
            "{path}.item_id must not be blank when present"
        )));
    }

    let arguments = call.arguments_value().map_err(|source| {
        Error::invalid_request(format!(
            "{path}.arguments must contain valid JSON: {source}"
        ))
        .with_source(source)
    })?;
    if !arguments.is_object() {
        return Err(Error::invalid_request(format!(
            "{path}.arguments must be a JSON object"
        )));
    }
    Ok(())
}

fn validate_tool_result(result: &ToolResultPart, path: &str) -> Result<()> {
    if result.call_id.trim().is_empty() {
        return Err(Error::invalid_request(format!(
            "{path}.call_id must not be blank"
        )));
    }
    for (field, value) in [
        ("item_id", result.item_id.as_deref()),
        ("name", result.name.as_deref()),
    ] {
        if value.is_some_and(|value| value.trim().is_empty()) {
            return Err(Error::invalid_request(format!(
                "{path}.{field} must not be blank when present"
            )));
        }
    }
    Ok(())
}

fn validate_matching_identity(result: &ToolResultPart, call: &ToolCall, path: &str) -> Result<()> {
    for (field, actual, expected) in [
        ("name", result.name.as_deref(), Some(call.name.as_str())),
        (
            "item_id",
            result.item_id.as_deref(),
            call.item_id.as_deref(),
        ),
    ] {
        if matches!((actual, expected), (Some(actual), Some(expected)) if actual != expected) {
            return Err(Error::invalid_request(format!(
                "{path}.{field} conflicts with the matching tool call"
            )));
        }
    }
    Ok(())
}

/// Reject what the model cannot do at all. Dropping these silently would
/// change the meaning of the request, unlike the sampling settings that
/// [`restrict_request`](super::restrict_request) clears with a warning.
fn validate_profile(
    request: &Request,
    profile: ApiProfile,
    capabilities: &ModelCapabilities,
) -> Result<()> {
    // The capabilities are the intersection of the profile and the model, so
    // the request may be refused on behalf of either.
    let unsupported = |what: &str| {
        Error::new(
            ErrorKind::UnsupportedCapability,
            format!("this model on `{profile}` does not support {what}"),
        )
    };
    if !capabilities.tools && !request.tools.is_empty() {
        return Err(unsupported("tool calls"));
    }
    if !capabilities.structured_output && request.structured_output.is_some() {
        return Err(unsupported("structured output"));
    }
    let requires_native_compaction = request.compaction.is_some()
        || request.messages.iter().any(|message| {
            matches!(message, Message::Assistant { content, .. }
            if content.iter().any(|part| matches!(
                part,
                AssistantPart::Compaction(compaction) if compaction.content.is_none()
            )))
        });
    if requires_native_compaction && !capabilities.native_compaction {
        return Err(unsupported("native compaction"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{CompactionPart, ToolResultContent};
    use crate::metadata::ProviderMetadata;
    use crate::request::{Compaction, StructuredOutput, ToolDefinition};

    type NumberSetter = fn(&mut Request, f64);

    fn request() -> Request {
        Request::builder().message(Message::user("hi")).build()
    }

    fn tool(name: &str) -> ToolDefinition {
        ToolDefinition::new(name, "description", serde_json::json!({"type": "object"}))
    }

    fn call(call_id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            call_id: call_id.into(),
            item_id: Some(format!("item_{call_id}")),
            name: name.into(),
            arguments: arguments.into(),
            provider_metadata: ProviderMetadata::default(),
        }
    }

    fn tool_result(call_id: &str) -> ToolResultPart {
        ToolResultPart {
            call_id: call_id.into(),
            item_id: Some(format!("item_{call_id}")),
            name: Some("lookup".into()),
            content: ToolResultContent::Text { text: "ok".into() },
            is_error: false,
        }
    }

    fn validate_request(request: &Request) -> Result<()> {
        validate(
            request,
            "model",
            ApiProfile::OpenAiChatCompletions,
            &ModelCapabilities::for_profile(ApiProfile::OpenAiChatCompletions),
        )
    }

    #[test]
    fn invalid_request_envelopes_are_rejected() {
        let empty_tool = Request::builder()
            .message(Message::Tool { content: vec![] })
            .build();
        let cases = [
            ("model identifier", request(), " "),
            ("no messages", Request::default(), "model"),
            ("tool result", empty_tool, "model"),
        ];

        for (expected, request, model) in cases {
            let error = validate(
                &request,
                model,
                ApiProfile::OpenAiChatCompletions,
                &ModelCapabilities::for_profile(ApiProfile::OpenAiChatCompletions),
            )
            .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidRequest);
            assert!(error.message().contains(expected), "{error}");
        }
    }

    #[test]
    fn empty_user_and_assistant_messages_are_accepted() {
        let requests = [
            Request::builder()
                .message(Message::User { content: vec![] })
                .build(),
            Request::builder()
                .message(Message::Assistant {
                    content: vec![],
                    provider_metadata: ProviderMetadata::default(),
                })
                .build(),
        ];

        for request in requests {
            assert!(validate_request(&request).is_ok());
        }
    }

    #[test]
    fn non_finite_numbers_are_rejected() {
        let setters: [(&str, NumberSetter); 4] = [
            ("temperature", |request, value| {
                request.temperature = Some(value)
            }),
            ("top_p", |request, value| request.top_p = Some(value)),
            ("presence_penalty", |request, value| {
                request.presence_penalty = Some(value);
            }),
            ("frequency_penalty", |request, value| {
                request.frequency_penalty = Some(value);
            }),
        ];

        for (name, set) in setters {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut request = request();
                set(&mut request, value);
                let error = validate_request(&request).unwrap_err();
                assert!(error.message().contains(name), "{error}");
            }
        }
    }

    #[test]
    fn zero_positive_counts_are_rejected() {
        let mut zero_output = request();
        zero_output.max_output_tokens = Some(0);
        let mut zero_top_k = request();
        zero_top_k.top_k = Some(0);

        for (expected, request) in [("max_output_tokens", zero_output), ("top_k", zero_top_k)] {
            let error = validate_request(&request).unwrap_err();
            assert!(error.message().contains(expected), "{error}");
        }
    }

    #[test]
    fn invalid_tool_definitions_are_rejected() {
        let mut blank = request();
        blank.tools.push(tool(" "));
        let mut duplicate = request();
        duplicate.tools = vec![tool("lookup"), tool("lookup")];
        let mut non_object = request();
        non_object.tools.push(ToolDefinition::new(
            "lookup",
            "description",
            serde_json::json!([]),
        ));

        for (expected, request) in [
            ("name", blank),
            ("declared more than once", duplicate),
            ("parameters", non_object),
        ] {
            let error = validate_request(&request).unwrap_err();
            assert!(error.message().contains(expected), "{error}");
        }

        let mut boolean_schema = request();
        boolean_schema.tools.push(ToolDefinition::new(
            "lookup",
            "description",
            serde_json::json!(true),
        ));
        assert!(validate_request(&boolean_schema).is_ok());
    }

    #[test]
    fn invalid_tool_choices_are_rejected() {
        let mut required = request();
        required.tool_choice = Some(ToolChoice::Required);
        let mut blank = request();
        blank.tools.push(tool("lookup"));
        blank.tool_choice = Some(ToolChoice::Tool { name: " ".into() });
        let mut undeclared = request();
        undeclared.tools.push(tool("lookup"));
        undeclared.tool_choice = Some(ToolChoice::Tool {
            name: "missing".into(),
        });

        for (expected, request) in [
            ("at least one", required),
            ("must not be blank", blank),
            ("undeclared", undeclared),
        ] {
            let error = validate_request(&request).unwrap_err();
            assert!(error.message().contains(expected), "{error}");
        }
    }

    #[test]
    fn provider_options_can_supply_the_tool_registry() {
        let requests = [
            (
                ApiProfile::OpenAiChatCompletions,
                Request::builder()
                    .message(Message::user("hi"))
                    .tool_choice(ToolChoice::Required)
                    .provider_option(
                        "openai",
                        serde_json::json!({"tools": [{"type": "web_search"}]}),
                    )
                    .build(),
            ),
            (
                ApiProfile::OpenAiChatCompletions,
                Request::builder()
                    .message(Message::user("hi"))
                    .tool_choice(ToolChoice::Tool {
                        name: "native_lookup".into(),
                    })
                    .provider_option(
                        "openai",
                        serde_json::json!({
                            "tools": [{
                                "type": "function",
                                "function": {"name": "native_lookup", "parameters": {}}
                            }]
                        }),
                    )
                    .build(),
            ),
            (
                ApiProfile::OpenAiChatCompletions,
                Request::builder()
                    .message(Message::user("hi"))
                    .tool_choice(ToolChoice::Required)
                    .provider_option("openai", serde_json::json!({"tool_choice": "none"}))
                    .build(),
            ),
            (
                ApiProfile::GeminiGenerateContent,
                Request::builder()
                    .message(Message::user("hi"))
                    .tool_choice(ToolChoice::Required)
                    .provider_option(
                        "gemini",
                        serde_json::json!({
                            "toolConfig": {"functionCallingConfig": {"mode": "NONE"}}
                        }),
                    )
                    .build(),
            ),
        ];

        for (profile, request) in requests {
            assert!(
                validate(
                    &request,
                    "model",
                    profile,
                    &ModelCapabilities::for_profile(profile)
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn structured_output_schema_must_be_an_object_or_boolean() {
        let mut request = request();
        request.structured_output = Some(StructuredOutput::new("result", serde_json::json!([])));

        let error = validate_request(&request).unwrap_err();

        assert!(error.message().contains("structured_output.schema"));

        request.structured_output = Some(StructuredOutput::new("result", serde_json::json!(true)));
        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn invalid_historical_tool_calls_are_rejected() {
        let mut blank_item_id = call("call_a", "lookup", "{}");
        blank_item_id.item_id = Some(" ".into());
        let cases = [
            ("call_id", call(" ", "lookup", "{}")),
            ("name", call("call_a", " ", "{}")),
            ("item_id", blank_item_id),
            ("valid JSON", call("call_a", "lookup", "{")),
            ("JSON object", call("call_a", "lookup", "[]")),
        ];
        for (expected, call) in cases {
            let request = Request::builder()
                .message(Message::Assistant {
                    content: vec![AssistantPart::ToolCall(call)],
                    provider_metadata: ProviderMetadata::default(),
                })
                .build();
            let error = validate_request(&request).unwrap_err();
            assert!(error.message().contains(expected), "{error}");
        }

        let duplicate = call("call_a", "lookup", "{}");
        let request = Request::builder()
            .message(Message::Assistant {
                content: vec![
                    AssistantPart::ToolCall(duplicate.clone()),
                    AssistantPart::ToolCall(duplicate),
                ],
                provider_metadata: ProviderMetadata::default(),
            })
            .build();
        let error = validate_request(&request).unwrap_err();
        assert!(error.message().contains("unresolved"), "{error}");
    }

    #[test]
    fn invalid_historical_tool_results_are_rejected() {
        let call = call("call_a", "lookup", "{}");
        let valid = tool_result("call_a");
        let mut blank = valid.clone();
        blank.call_id = " ".into();
        let mut blank_name = valid.clone();
        blank_name.name = Some(" ".into());
        let mut blank_item = valid.clone();
        blank_item.item_id = Some(" ".into());
        let mut wrong_name = valid.clone();
        wrong_name.name = Some("other".into());
        let mut wrong_item = valid.clone();
        wrong_item.item_id = Some("other".into());
        let cases = [
            ("call_id", vec![blank]),
            ("name", vec![blank_name]),
            ("item_id", vec![blank_item]),
            ("name", vec![wrong_name]),
            ("item_id", vec![wrong_item]),
            ("more than one result", vec![valid.clone(), valid]),
        ];

        for (expected, results) in cases {
            let request = Request::builder()
                .message(Message::Assistant {
                    content: vec![AssistantPart::ToolCall(call.clone())],
                    provider_metadata: ProviderMetadata::default(),
                })
                .message(Message::Tool { content: results })
                .build();
            let error = validate_request(&request).unwrap_err();
            assert!(error.message().contains(expected), "{error}");
        }
    }

    #[test]
    fn valid_tool_request_is_accepted() {
        let call = call("call_a", "lookup", r#"{"query":"value"}"#);
        let result = ToolResultPart::for_call(&call, "ok");
        let request = Request::builder()
            .message(Message::Assistant {
                content: vec![AssistantPart::ToolCall(call)],
                provider_metadata: ProviderMetadata::default(),
            })
            .message(Message::tool_result(result))
            .tool(tool("lookup"))
            .tool_choice(ToolChoice::Tool {
                name: "lookup".into(),
            })
            .structured_output(StructuredOutput::new(
                "result",
                serde_json::json!({"type": "object"}),
            ))
            .build();

        assert!(validate_request(&request).is_ok());

        let orphan_result = Request::builder()
            .message(Message::tool_result(tool_result("orphan")))
            .build();
        assert!(validate_request(&orphan_result).is_ok());
    }

    #[test]
    fn tool_call_ids_can_be_reused_after_resolution() {
        let first = call("call_0", "lookup", "{}");
        let second = call("call_0", "lookup", "{}");
        let request = Request::builder()
            .message(Message::Assistant {
                content: vec![AssistantPart::ToolCall(first.clone())],
                provider_metadata: ProviderMetadata::default(),
            })
            .message(Message::tool_result(ToolResultPart::for_call(
                &first, "first",
            )))
            .message(Message::Assistant {
                content: vec![AssistantPart::ToolCall(second.clone())],
                provider_metadata: ProviderMetadata::default(),
            })
            .message(Message::tool_result(ToolResultPart::for_call(
                &second, "second",
            )))
            .build();

        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn result_before_same_id_call_remains_self_contained() {
        let mut result = tool_result("call_0");
        result.name = Some("external_lookup".into());
        result.item_id = Some("external_item".into());
        let request = Request::builder()
            .message(Message::tool_result(result))
            .message(Message::Assistant {
                content: vec![AssistantPart::ToolCall(call(
                    "call_0",
                    "local_lookup",
                    "{}",
                ))],
                provider_metadata: ProviderMetadata::default(),
            })
            .build();

        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn validation_errors_have_profile_and_model_context() {
        let mut request = request();
        request.max_output_tokens = Some(0);

        let error = validate(
            &request,
            "model",
            ApiProfile::OpenAiChatCompletions,
            &ModelCapabilities::for_profile(ApiProfile::OpenAiChatCompletions),
        )
        .unwrap_err();

        assert_eq!(
            error.origin(),
            Some(ApiProfile::OpenAiChatCompletions.as_str())
        );
        assert_eq!(error.model(), Some("model"));
    }

    #[test]
    fn native_compaction_is_rejected_by_chat_protocol() {
        let request = Request::builder()
            .message(Message::user("hi"))
            .compaction(Compaction::enabled())
            .build();

        let error = validate_request(&request).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
    }

    #[test]
    fn native_compaction_is_accepted_by_chatgpt_responses() {
        let request = Request::builder()
            .message(Message::user("hi"))
            .compaction(Compaction::enabled())
            .build();

        assert!(
            validate(
                &request,
                "model",
                ApiProfile::ChatGptResponses,
                &ModelCapabilities::for_profile(ApiProfile::ChatGptResponses)
            )
            .is_ok()
        );
    }

    #[test]
    fn opaque_compaction_history_is_rejected_by_chat_protocol() {
        let request = Request::builder()
            .message(Message::Assistant {
                content: vec![AssistantPart::Compaction(CompactionPart {
                    id: None,
                    content: None,
                    encrypted_content: Some("opaque".into()),
                })],
                provider_metadata: ProviderMetadata::default(),
            })
            .build();

        let error = validate_request(&request).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
    }
}
