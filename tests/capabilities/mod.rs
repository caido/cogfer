//! [`ModelCapabilities`] must describe what lowering actually sends.
//!
//! The profile tables are hand-maintained, so every profile is exercised
//! with every request setting and the dropped ones are compared against the
//! table: a `false` capability must produce exactly one unsupported-setting
//! warning and a `true` one must produce none.

use std::collections::BTreeSet;
use std::num::NonZeroU32;

use llmwire::transport::mock::MockTransport;
use llmwire::{
    ApiProfile, Compaction, Credentials, ErrorKind, GenerateResult, Message, ModelCapabilities,
    ProviderConfig, ReasoningConfig, ReasoningEffort, ReasoningOutput, Request, ToolDefinition,
    WarningKind,
};
use serde_json::{Value, json};

use crate::common::provider_with;

/// A slice rather than an array so the Bedrock entry can be conditional
/// without restating the whole list per feature combination.
const PROFILES: &[ApiProfile] = &[
    ApiProfile::OpenAiResponses,
    ApiProfile::OpenAiChatCompletions,
    ApiProfile::OpenRouter,
    ApiProfile::ChatGptResponses,
    ApiProfile::XaiResponses,
    ApiProfile::XaiChatCompletions,
    ApiProfile::AnthropicMessages,
    #[cfg(feature = "aws")]
    ApiProfile::BedrockAnthropic,
    #[cfg(feature = "aws")]
    ApiProfile::BedrockOpenAiResponses,
    ApiProfile::GeminiGenerateContent,
];

fn config(profile: ApiProfile) -> ProviderConfig {
    ProviderConfig::new(profile, Credentials::api_key("k"))
}

/// Queue one successful reply in the profile's wire format.
fn queue_success(mock: &MockTransport, profile: ApiProfile) {
    match profile {
        #[cfg(feature = "aws")]
        ApiProfile::BedrockOpenAiResponses => mock.push_json(
            200,
            &json!({
                "id": "resp_1", "object": "response", "status": "completed", "model": "m",
                "output": [{"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            }),
        ),
        ApiProfile::OpenAiResponses | ApiProfile::XaiResponses => mock.push_json(
            200,
            &json!({
                "id": "resp_1", "object": "response", "status": "completed", "model": "m",
                "output": [{"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            }),
        ),
        ApiProfile::ChatGptResponses => mock.push_sse(&[
            r#"{"type":"response.created","response":{"id":"resp_1","model":"m","status":"in_progress"}}"#,
            r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok","annotations":[]}]}}"#,
            r#"{"type":"response.completed","response":{"id":"resp_1","status":"completed","model":"m","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok","annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}"#,
            "[DONE]",
        ]),
        ApiProfile::OpenAiChatCompletions | ApiProfile::OpenRouter | ApiProfile::XaiChatCompletions => {
            mock.push_json(
                200,
                &json!({
                    "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }),
            );
        }
        #[cfg(feature = "aws")]
        ApiProfile::BedrockAnthropic => mock.push_json(
            200,
            &json!({
                "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn", "stop_sequence": null,
                "usage": {"input_tokens": 1, "output_tokens": 1}
            }),
        ),
        ApiProfile::AnthropicMessages => mock.push_json(
            200,
            &json!({
                "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn", "stop_sequence": null,
                "usage": {"input_tokens": 1, "output_tokens": 1}
            }),
        ),
        ApiProfile::GeminiGenerateContent => mock.push_json(
            200,
            &json!({
                "candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"},
                                "finishReason": "STOP", "index": 0}],
                "modelVersion": "m", "responseId": "r1",
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
            }),
        ),
        _ => unreachable!("every profile has a canned reply"),
    }
}

/// Lower `request` on `profile` and return the result plus the wire body.
async fn lower(profile: ApiProfile, request: Request) -> (GenerateResult, Value) {
    let mock = MockTransport::shared();
    queue_success(&mock, profile);
    let result = provider_with(&mock, config(profile))
        .language_model("m")
        .generate(request)
        .await
        .unwrap_or_else(|error| panic!("{profile}: generate fails: {error}"));
    (result, mock.request_json(0))
}

/// A request with every setting the capabilities table describes, except
/// compaction (an error, not a warning, where unsupported) and a reasoning
/// budget (reasoning effort exercises reasoning support instead).
fn everything() -> Request {
    let mut tool = ToolDefinition::new("lookup", "Look something up", json!({"type": "object"}));
    tool.strict = Some(true);
    Request::builder()
        .message(Message::user("hi"))
        .tool(tool)
        .parallel_tool_calls(true)
        .reasoning(ReasoningConfig::Effort {
            effort: ReasoningEffort::Medium,
            output: Some(ReasoningOutput::Include),
        })
        .max_output_tokens(256)
        .temperature(0.5)
        .top_p(0.9)
        .top_k(40)
        .stop_sequence("END")
        .seed(7)
        .presence_penalty(0.1)
        .frequency_penalty(0.1)
        .build()
}

fn expected_dropped(capabilities: &ModelCapabilities) -> BTreeSet<&'static str> {
    [
        ("tools", capabilities.tools),
        ("structured_output", capabilities.structured_output),
        ("tools.strict", capabilities.strict_tools),
        ("parallel_tool_calls", capabilities.parallel_tool_calls),
        ("reasoning.output", capabilities.reasoning.output),
        ("max_output_tokens", capabilities.max_output_tokens),
        ("temperature", capabilities.temperature),
        ("top_p", capabilities.top_p),
        ("top_k", capabilities.top_k),
        ("stop_sequences", capabilities.stop_sequences),
        ("seed", capabilities.seed),
        ("presence_penalty", capabilities.presence_penalty),
        ("frequency_penalty", capabilities.frequency_penalty),
    ]
    .into_iter()
    .filter(|(_, supported)| !supported)
    .map(|(subject, _)| subject)
    .collect()
}

#[tokio::test]
async fn unsupported_settings_match_the_capability_table() {
    for &profile in PROFILES {
        let capabilities = ModelCapabilities::for_profile(profile);
        let (result, _) = lower(profile, everything()).await;

        let dropped: BTreeSet<&str> = result
            .warnings
            .iter()
            .filter(|warning| warning.kind == WarningKind::UnsupportedSetting)
            .map(|warning| warning.subject.as_deref().expect("subject"))
            .collect();
        let mut expected = expected_dropped(&capabilities);
        // The library omits sampling settings for Anthropic when thinking is
        // enabled, including adaptive mode.
        #[cfg(feature = "aws")]
        let anthropic_thinking = matches!(
            profile,
            ApiProfile::AnthropicMessages | ApiProfile::BedrockAnthropic
        );
        #[cfg(not(feature = "aws"))]
        let anthropic_thinking = matches!(profile, ApiProfile::AnthropicMessages);
        if anthropic_thinking {
            expected.extend(["temperature", "top_p", "top_k"]);
        }
        assert_eq!(dropped, expected, "{profile}: {:?}", result.warnings);
        assert!(
            !result
                .warnings
                .iter()
                .any(|warning| warning.kind == WarningKind::ApproximatedSetting),
            "{profile}: medium effort is native everywhere: {:?}",
            result.warnings
        );
    }
}

#[tokio::test]
async fn native_compaction_matches_the_capability_table() {
    for &profile in PROFILES {
        let request = Request::builder()
            .message(Message::user("hi"))
            .compaction(Compaction::enabled())
            .build();
        let mock = MockTransport::shared();
        queue_success(&mock, profile);
        let outcome = provider_with(&mock, config(profile))
            .language_model("m")
            .generate(request)
            .await;

        match outcome {
            Ok(_) => assert!(
                ModelCapabilities::for_profile(profile).native_compaction,
                "{profile} accepted compaction but the table says otherwise"
            ),
            Err(error) => {
                assert_eq!(error.kind(), ErrorKind::UnsupportedCapability, "{profile}");
                assert!(
                    !ModelCapabilities::for_profile(profile).native_compaction,
                    "{profile} rejected compaction but the table says otherwise"
                );
            }
        }
    }
}

#[tokio::test]
async fn every_listed_effort_is_sent_verbatim() {
    for &profile in PROFILES {
        for effort in ModelCapabilities::for_profile(profile).reasoning.efforts {
            let request = Request::builder()
                .message(Message::user("hi"))
                .reasoning(ReasoningConfig::effort(effort))
                .build();
            let (result, body) = lower(profile, request).await;

            assert!(
                result.warnings.iter().all(|warning| !warning
                    .subject
                    .as_deref()
                    .is_some_and(|subject| subject.starts_with("reasoning"))),
                "{profile} {effort:?}: {:?}",
                result.warnings
            );
            assert!(
                body.to_string()
                    .contains(&format!("\"{}\"", effort.as_str())),
                "{profile} {effort:?}: {body}"
            );
        }
    }
}

fn approximations(result: &GenerateResult) -> Vec<&str> {
    result
        .warnings
        .iter()
        .filter(|warning| warning.kind == WarningKind::ApproximatedSetting)
        .map(|warning| warning.subject.as_deref().unwrap_or_default())
        .collect()
}

#[tokio::test]
async fn budgets_are_approximated_as_efforts_where_only_efforts_exist() {
    let request = Request::builder()
        .message(Message::user("hi"))
        .reasoning(ReasoningConfig::budget(NonZeroU32::new(20_000).unwrap()))
        .build();

    let (result, body) = lower(ApiProfile::OpenAiChatCompletions, request.clone()).await;
    assert_eq!(body["reasoning_effort"], "medium", "{body}");
    assert_eq!(approximations(&result), ["reasoning.budget"]);

    let (result, body) = lower(ApiProfile::OpenAiResponses, request.clone()).await;
    assert_eq!(body["reasoning"]["effort"], "medium", "{body}");
    assert_eq!(approximations(&result), ["reasoning.budget"]);

    // Budgets are native here, so nothing is approximated.
    let (result, body) = lower(ApiProfile::OpenRouter, request).await;
    assert_eq!(body["reasoning"]["max_tokens"], 20_000, "{body}");
    assert!(approximations(&result).is_empty(), "{:?}", result.warnings);
}

#[tokio::test]
async fn out_of_range_efforts_are_clamped_to_the_nearest_supported_level() {
    let effort = |effort| {
        Request::builder()
            .message(Message::user("hi"))
            .reasoning(ReasoningConfig::effort(effort))
            .build()
    };

    // Anthropic has no `minimal`, so it snaps to `low`.
    let (result, body) = lower(
        ApiProfile::AnthropicMessages,
        effort(ReasoningEffort::Minimal),
    )
    .await;
    assert_eq!(body["output_config"]["effort"], "low", "{body}");
    assert_eq!(approximations(&result), ["reasoning.effort"]);

    // Gemini tops out at `high`.
    let (result, body) = lower(
        ApiProfile::GeminiGenerateContent,
        effort(ReasoningEffort::Max),
    )
    .await;
    assert_eq!(
        body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "high",
        "{body}"
    );
    assert_eq!(approximations(&result), ["reasoning.effort"]);

    // SpaceXAI tops out at `xhigh`.
    let (result, body) = lower(ApiProfile::XaiResponses, effort(ReasoningEffort::Max)).await;
    assert_eq!(body["reasoning"]["effort"], "xhigh", "{body}");
    assert_eq!(approximations(&result), ["reasoning.effort"]);
}

#[test]
fn models_expose_their_provider_capabilities() {
    let mock = MockTransport::shared();
    let model = provider_with(&mock, config(ApiProfile::AnthropicMessages)).language_model("m");

    let capabilities = model.capabilities();

    assert_eq!(
        *capabilities,
        ModelCapabilities::for_profile(ApiProfile::AnthropicMessages)
    );
    assert!(capabilities.reasoning.budget);
    assert!(!capabilities.seed);
}

#[tokio::test]
async fn model_capabilities_narrow_the_profile_defaults() {
    let profile = ApiProfile::OpenAiChatCompletions;
    let model_data = ModelCapabilities {
        temperature: false,
        reasoning: llmwire::ReasoningSupport {
            efforts: vec![ReasoningEffort::Low],
            ..ModelCapabilities::for_profile(profile).reasoning
        },
        ..ModelCapabilities::for_profile(profile)
    };
    let mock = MockTransport::shared();
    queue_success(&mock, profile);
    let model = provider_with(&mock, config(profile))
        .language_model("m")
        .with_capabilities(&model_data);
    assert!(!model.capabilities().temperature);
    assert_eq!(
        model.capabilities().reasoning.efforts,
        [ReasoningEffort::Low]
    );

    let result = model
        .generate(
            Request::builder()
                .message(Message::user("hi"))
                .temperature(0.3)
                .reasoning(ReasoningConfig::effort(ReasoningEffort::High))
                .build(),
        )
        .await
        .expect("generate succeeds");

    let body = mock.request_json(0);
    assert!(body.get("temperature").is_none(), "{body}");
    assert_eq!(body["reasoning_effort"], "low", "{body}");
    let subjects: Vec<_> = result
        .warnings
        .iter()
        .map(|warning| (warning.kind, warning.subject.as_deref().unwrap()))
        .collect();
    assert_eq!(
        subjects,
        [
            (WarningKind::UnsupportedSetting, "temperature"),
            (WarningKind::ApproximatedSetting, "reasoning.effort"),
        ]
    );
}

#[tokio::test]
async fn models_without_tool_support_reject_tool_requests() {
    let profile = ApiProfile::OpenAiChatCompletions;
    let model_data = ModelCapabilities {
        tools: false,
        ..ModelCapabilities::for_profile(profile)
    };
    let mock = MockTransport::shared();
    let model = provider_with(&mock, config(profile))
        .language_model("m")
        .with_capabilities(&model_data);

    let error = model
        .generate(everything())
        .await
        .expect_err("tools cannot be dropped silently");

    assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
    assert!(mock.requests().is_empty());
}
