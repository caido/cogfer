//! Wire fragments shared by the OpenAI Chat Completions and Responses formats.

use serde_json::{Map, Value, json};

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
