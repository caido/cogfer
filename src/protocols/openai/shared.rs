//! Wire fragments shared by the Chat Completions and Responses formats.

use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::error::{Error, ErrorKind};
use crate::http::{enrich_error_from_headers, error_kind_for_status};
use crate::protocols::{ApiProfile, content_policy_kind, fallback_provider_error_message};
use crate::request::{Request, StructuredOutput, ToolChoice, ToolDefinition};

/// How the two formats nest function definitions and forced tool choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolShape {
    /// Chat Completions wraps functions in `{"type": "function", "function": {..}}`.
    Nested,
    /// Responses puts the function fields on the tool object itself.
    Flat,
}

fn function_tool(tool: &ToolDefinition, shape: ToolShape) -> Value {
    let mut function = json!({
        "name": tool.name,
        "parameters": tool.parameters,
    });
    if let Some(description) = &tool.description {
        function["description"] = json!(description);
    }
    if let Some(strict) = tool.strict {
        function["strict"] = json!(strict);
    }
    match shape {
        ToolShape::Nested => json!({"type": "function", "function": function}),
        ToolShape::Flat => {
            function["type"] = json!("function");
            function
        }
    }
}

fn tool_choice(choice: &ToolChoice, shape: ToolShape) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool { name } => match shape {
            ToolShape::Nested => json!({"type": "function", "function": {"name": name}}),
            ToolShape::Flat => json!({"type": "function", "name": name}),
        },
    }
}

/// Insert `tools`, `tool_choice`, and `parallel_tool_calls` when set.
pub(crate) fn insert_tools(request: &Request, object: &mut Map<String, Value>, shape: ToolShape) {
    if !request.tools.is_empty() {
        let tools = request
            .tools
            .iter()
            .map(|tool| function_tool(tool, shape))
            .collect();
        object.insert("tools".into(), Value::Array(tools));
    }
    if let Some(choice) = &request.tool_choice {
        object.insert("tool_choice".into(), tool_choice(choice, shape));
    }
    if let Some(parallel) = request.parallel_tool_calls {
        object.insert("parallel_tool_calls".into(), json!(parallel));
    }
}

/// The `json_schema` structured-output configuration shared by both formats.
pub(crate) fn json_schema_format(output: &StructuredOutput) -> Value {
    let mut format = json!({
        "name": output.name,
        "schema": output.schema,
        "strict": output.strict,
    });
    if let Some(description) = &output.description {
        format["description"] = json!(description);
    }
    format
}

/// Decode an OpenAI-style error envelope shared by Responses and Chat.
pub(crate) fn decode_openai_error(
    protocol: ApiProfile,
    status: u16,
    headers: &[(String, String)],
    body: &[u8],
) -> Error {
    #[derive(Deserialize)]
    struct Envelope {
        error: Option<ErrorBody>,
    }
    #[derive(Deserialize)]
    struct ErrorBody {
        message: Option<String>,
        #[serde(rename = "type")]
        error_type: Option<String>,
        code: Option<Value>,
    }

    let parsed = serde_json::from_slice::<Envelope>(body)
        .ok()
        .and_then(|envelope| envelope.error);
    let code = parsed.as_ref().and_then(|error| match &error.code {
        Some(Value::String(code)) => Some(code.clone()),
        Some(Value::Number(code)) => Some(code.to_string()),
        _ => None,
    });
    let message = parsed.as_ref().and_then(|error| error.message.clone());

    let mut kind = error_kind_for_status(status);
    if matches!(
        code.as_deref(),
        Some("context_length_exceeded" | "string_above_max_length")
    ) {
        kind = ErrorKind::ContextLength;
    }
    // OpenAI answers exhausted billing with a 429, and retrying cannot help.
    if code.as_deref() == Some("insufficient_quota") {
        kind = ErrorKind::Permission;
    }
    if parsed
        .as_ref()
        .and_then(|error| error.error_type.as_deref())
        .is_some_and(|error_type| error_type.contains("authentication"))
    {
        kind = ErrorKind::Authentication;
    }
    kind = content_policy_kind(code.as_deref(), kind);

    let message =
        message.unwrap_or_else(|| fallback_provider_error_message(protocol, status, body));
    let mut error = Error::new(kind, message).with_origin(protocol.as_str());
    if let Some(code) = code {
        error = error.with_code(code);
    }
    enrich_error_from_headers(error, status, headers)
}
