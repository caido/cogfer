//! Multi-turn flows shared by the live tests and the replay scenarios.

use cogfer::{FinishReason, LanguageModel, Message, Request, ToolDefinition, ToolResultPart};
use serde_json::json;

use super::{assert_terminal_contract, collect, drain};

pub(crate) fn time_tool() -> ToolDefinition {
    ToolDefinition::new(
        "get_current_time",
        "Returns the current time for a city.",
        json!({
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
            "additionalProperties": false
        }),
    )
}

/// Drive a two-turn agentic tool loop: the model calls the tool, we answer,
/// the model produces the final text. Validates tool identity and reasoning
/// replay end to end.
pub(crate) async fn agentic_loop(model: &LanguageModel, base: Request) {
    let mut messages = base.messages.clone();

    // Turn 1 is streamed so the stream contract is exercised.
    let events = drain(
        model
            .stream(base.clone())
            .await
            .expect("turn 1 stream establishes"),
    )
    .await;
    assert_terminal_contract(&events);
    let result = collect(&events).expect("turn 1 result");
    assert_eq!(
        result.finish.reason,
        FinishReason::ToolCalls,
        "model should call the tool; got content: {:?}",
        result.content
    );
    let call = result.tool_calls().next().expect("one tool call").clone();
    assert_eq!(call.name, "get_current_time");
    let arguments: serde_json::Value = call.parse_arguments().expect("valid arguments JSON");
    assert!(arguments.get("city").is_some(), "arguments: {arguments}");

    messages.push(result.to_assistant_message());
    messages.push(Message::tool_result(ToolResultPart::for_call(
        &call,
        "It is exactly 12:00 noon.",
    )));

    let mut request = base.clone();
    request.messages = messages;
    let final_result = model.generate(request).await.expect("turn 2 succeeds");
    let text = final_result.text().to_lowercase();
    assert!(
        text.contains("12") || text.contains("noon"),
        "final answer should use the tool result: {text:?}"
    );
    assert!(final_result.usage.total_input_tokens().is_some());
}

pub(crate) fn loop_request() -> Request {
    Request::builder()
        .system("You are a helpful assistant. Use the get_current_time tool to answer time questions; afterwards answer from its result.")
        .message(Message::user("What time is it in Paris right now?"))
        .tool(time_tool())
        .max_output_tokens(3000)
        .build()
}
