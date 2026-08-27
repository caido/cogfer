//! Fit a request to what the model accepts before lowering.
//!
//! Settings the [`ModelCapabilities`] table marks unsupported are cleared
//! here, once, with an [`UnsupportedSetting`](crate::WarningKind::UnsupportedSetting)
//! warning each, so protocol lowering only sees what it can send. Reasoning
//! is the exception: it maps onto a different control rather than being
//! dropped, which [`resolve`](super::reasoning::resolve) handles per profile.

use super::ApiProfile;
use crate::capabilities::ModelCapabilities;
use crate::request::Request;
use crate::response::Warning;

/// A copy of `request` without the settings `capabilities` rules out, plus
/// a warning for each one removed.
pub(crate) fn restrict_request(
    request: &Request,
    capabilities: &ModelCapabilities,
    profile: ApiProfile,
) -> (Request, Vec<Warning>) {
    let mut request = request.clone();
    let mut warnings = Vec::new();
    let mut drop = |setting: &str| {
        warnings.push(Warning::unsupported_setting(
            setting,
            format!("{profile} does not support `{setting}`"),
        ));
    };

    if !capabilities.max_output_tokens && request.max_output_tokens.take().is_some() {
        drop("max_output_tokens");
    }
    if !capabilities.temperature && request.temperature.take().is_some() {
        drop("temperature");
    }
    if !capabilities.top_p && request.top_p.take().is_some() {
        drop("top_p");
    }
    if !capabilities.top_k && request.top_k.take().is_some() {
        drop("top_k");
    }
    if !capabilities.stop_sequences && !request.stop_sequences.is_empty() {
        request.stop_sequences.clear();
        drop("stop_sequences");
    }
    if !capabilities.seed && request.seed.take().is_some() {
        drop("seed");
    }
    if !capabilities.presence_penalty && request.presence_penalty.take().is_some() {
        drop("presence_penalty");
    }
    if !capabilities.frequency_penalty && request.frequency_penalty.take().is_some() {
        drop("frequency_penalty");
    }
    if !capabilities.parallel_tool_calls && request.parallel_tool_calls.take().is_some() {
        drop("parallel_tool_calls");
    }
    if !capabilities.strict_tools && request.tools.iter().any(|tool| tool.strict.is_some()) {
        for tool in &mut request.tools {
            tool.strict = None;
        }
        drop("tools.strict");
    }
    (request, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    use crate::request::ToolDefinition;

    #[test]
    fn unsupported_settings_are_cleared_with_one_warning_each() {
        let mut capabilities = ModelCapabilities::for_profile(ApiProfile::OpenAiChatCompletions);
        capabilities.temperature = false;
        capabilities.strict_tools = false;
        let mut tool = ToolDefinition::new("t", "tool", serde_json::json!({}));
        tool.strict = Some(true);
        let request = Request::builder()
            .message(Message::user("hi"))
            .temperature(0.2)
            .top_p(0.9)
            .tool(tool)
            .build();

        let (restricted, warnings) =
            restrict_request(&request, &capabilities, ApiProfile::OpenAiChatCompletions);

        assert_eq!(restricted.temperature, None);
        assert_eq!(restricted.top_p, Some(0.9));
        assert_eq!(restricted.tools[0].strict, None);
        let subjects: Vec<_> = warnings
            .iter()
            .map(|warning| warning.subject.as_deref().unwrap())
            .collect();
        assert_eq!(subjects, ["temperature", "tools.strict"]);
    }

    #[test]
    fn supported_settings_pass_through_untouched() {
        let capabilities = ModelCapabilities::for_profile(ApiProfile::OpenRouter);
        let request = Request::builder()
            .message(Message::user("hi"))
            .top_k(4)
            .seed(1)
            .build();

        let (restricted, warnings) =
            restrict_request(&request, &capabilities, ApiProfile::OpenRouter);

        assert_eq!(restricted, request);
        assert!(warnings.is_empty());
    }
}
