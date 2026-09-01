//! What a model can do on its API profile.
//!
//! Every [`Request`](crate::Request) setting is expressible, but not every
//! model accepts every setting. Requests are fitted to the model's
//! capabilities before lowering: most unsupported settings are dropped with
//! a [`Warning`](crate::Warning), reasoning is mapped onto the closest
//! control, and tools, structured output, and compaction fail the request.
//! These types say up front what will survive so hosts can hide or disable
//! the controls that would not.
//!
//! [`ModelCapabilities::for_profile`] gives the profile defaults. Per-model
//! data (a models.dev listing, for example) narrows them through
//! [`LanguageModel::with_capabilities`](crate::LanguageModel::with_capabilities).

use std::num::NonZeroU32;

use crate::protocols::ApiProfile;
use crate::request::ReasoningEffort;

/// How a profile controls reasoning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningSupport {
    /// The discrete efforts accepted on the wire, from least to most. Empty
    /// when the profile has no effort control.
    pub efforts: Vec<ReasoningEffort>,
    /// Whether an exact reasoning-token budget can be sent.
    pub budget: bool,
    /// Whether reasoning can be switched off.
    pub disable: bool,
    /// Whether the visibility of reasoning in the response can be controlled.
    pub output: bool,
}

impl ReasoningSupport {
    /// A profile without any reasoning control.
    pub fn none() -> Self {
        Self {
            efforts: Vec::new(),
            budget: false,
            disable: false,
            output: false,
        }
    }

    /// The supported effort closest to `effort`, preferring the lower one
    /// when `effort` falls between two.
    pub(crate) fn nearest_effort(&self, effort: ReasoningEffort) -> Option<ReasoningEffort> {
        if self.efforts.contains(&effort) {
            return Some(effort);
        }
        let rank = |candidate: ReasoningEffort| candidate as i32;
        self.efforts.iter().copied().min_by_key(|candidate| {
            let distance = (rank(*candidate) - rank(effort)).abs();
            (distance, rank(*candidate))
        })
    }

    /// The supported effort that approximates a token budget.
    pub(crate) fn effort_for_budget(&self, tokens: NonZeroU32) -> Option<ReasoningEffort> {
        let effort = match tokens.get() {
            0..1024 => ReasoningEffort::Minimal,
            1024..8192 => ReasoningEffort::Low,
            8192..32768 => ReasoningEffort::Medium,
            32768.. => ReasoningEffort::High,
        };
        self.nearest_effort(effort)
    }

    /// The controls both `self` and `other` offer.
    pub fn restrict(&self, other: &Self) -> Self {
        Self {
            efforts: self
                .efforts
                .iter()
                .copied()
                .filter(|effort| other.efforts.contains(effort))
                .collect(),
            budget: self.budget && other.budget,
            disable: self.disable && other.disable,
            output: self.output && other.output,
        }
    }

    /// The token budget that approximates a discrete effort.
    pub(crate) fn budget_for_effort(effort: ReasoningEffort) -> NonZeroU32 {
        let tokens = match effort {
            ReasoningEffort::Minimal => 1024,
            ReasoningEffort::Low => 4096,
            ReasoningEffort::Medium => 16384,
            ReasoningEffort::High => 32768,
            ReasoningEffort::XHigh => 65536,
            ReasoningEffort::Max => 131_072,
        };
        NonZeroU32::new(tokens).expect("budgets are non-zero")
    }
}

/// The settings a model accepts.
///
/// Build a model-specific value from the profile defaults with struct
/// update syntax, `ModelCapabilities { seed: false, ..ModelCapabilities::for_profile(profile) }`,
/// and hand it to [`LanguageModel::with_capabilities`](crate::LanguageModel::with_capabilities).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub reasoning: ReasoningSupport,
    pub tools: bool,
    /// Provider-side strict tool-schema validation ([`ToolDefinition::strict`](crate::ToolDefinition::strict)).
    pub strict_tools: bool,
    pub parallel_tool_calls: bool,
    pub structured_output: bool,
    /// Provider-managed context compaction ([`Request::compaction`](crate::Request::compaction)).
    /// Unlike the other settings, a request that needs it fails with
    /// [`ErrorKind::UnsupportedCapability`](crate::ErrorKind::UnsupportedCapability)
    /// instead of a warning.
    pub native_compaction: bool,
    pub max_output_tokens: bool,
    pub temperature: bool,
    pub top_p: bool,
    pub top_k: bool,
    pub stop_sequences: bool,
    pub seed: bool,
    pub presence_penalty: bool,
    pub frequency_penalty: bool,
}

impl ModelCapabilities {
    /// The settings both `self` and `other` accept.
    pub fn restrict(&self, other: &Self) -> Self {
        Self {
            reasoning: self.reasoning.restrict(&other.reasoning),
            tools: self.tools && other.tools,
            strict_tools: self.strict_tools && other.strict_tools,
            parallel_tool_calls: self.parallel_tool_calls && other.parallel_tool_calls,
            structured_output: self.structured_output && other.structured_output,
            native_compaction: self.native_compaction && other.native_compaction,
            max_output_tokens: self.max_output_tokens && other.max_output_tokens,
            temperature: self.temperature && other.temperature,
            top_p: self.top_p && other.top_p,
            top_k: self.top_k && other.top_k,
            stop_sequences: self.stop_sequences && other.stop_sequences,
            seed: self.seed && other.seed,
            presence_penalty: self.presence_penalty && other.presence_penalty,
            frequency_penalty: self.frequency_penalty && other.frequency_penalty,
        }
    }

    /// The defaults for every model on `profile`.
    pub fn for_profile(profile: ApiProfile) -> Self {
        use ReasoningEffort::{High, Low, Max, Medium, Minimal, XHigh};

        let openai_reasoning = |output: bool| ReasoningSupport {
            efforts: vec![Minimal, Low, Medium, High, XHigh],
            budget: false,
            disable: true,
            output,
        };
        let full = Self {
            reasoning: ReasoningSupport::none(),
            tools: true,
            strict_tools: true,
            parallel_tool_calls: true,
            structured_output: true,
            native_compaction: false,
            max_output_tokens: true,
            temperature: true,
            top_p: true,
            top_k: true,
            stop_sequences: true,
            seed: true,
            presence_penalty: true,
            frequency_penalty: true,
        };
        match profile {
            ApiProfile::OpenAiResponses => Self {
                reasoning: openai_reasoning(true),
                native_compaction: true,
                top_k: false,
                stop_sequences: false,
                seed: false,
                presence_penalty: false,
                frequency_penalty: false,
                ..full
            },
            #[cfg(feature = "aws")]
            ApiProfile::BedrockOpenAiResponses => Self {
                reasoning: openai_reasoning(true),
                top_k: false,
                stop_sequences: false,
                seed: false,
                presence_penalty: false,
                frequency_penalty: false,
                ..full
            },
            ApiProfile::ChatGptResponses => Self {
                reasoning: openai_reasoning(true),
                native_compaction: true,
                max_output_tokens: false,
                temperature: false,
                top_p: false,
                top_k: false,
                stop_sequences: false,
                seed: false,
                presence_penalty: false,
                frequency_penalty: false,
                ..full
            },
            ApiProfile::OpenAiChatCompletions => Self {
                reasoning: openai_reasoning(false),
                top_k: false,
                ..full
            },
            ApiProfile::XaiResponses => Self {
                reasoning: ReasoningSupport {
                    efforts: vec![Low, Medium, High],
                    budget: false,
                    disable: true,
                    output: true,
                },
                top_k: false,
                stop_sequences: false,
                seed: false,
                presence_penalty: false,
                frequency_penalty: false,
                ..full
            },
            ApiProfile::XaiChatCompletions => Self {
                reasoning: ReasoningSupport {
                    efforts: vec![Low, Medium, High],
                    budget: false,
                    disable: true,
                    output: false,
                },
                top_k: false,
                ..full
            },
            ApiProfile::OpenRouter => Self {
                reasoning: ReasoningSupport {
                    efforts: vec![Minimal, Low, Medium, High, XHigh],
                    budget: true,
                    disable: true,
                    output: true,
                },
                ..full
            },
            ApiProfile::AnthropicMessages => Self {
                reasoning: ReasoningSupport {
                    efforts: vec![Low, Medium, High, XHigh, Max],
                    budget: true,
                    disable: true,
                    output: true,
                },
                native_compaction: true,
                seed: false,
                presence_penalty: false,
                frequency_penalty: false,
                ..full
            },
            #[cfg(feature = "aws")]
            ApiProfile::BedrockAnthropic => Self {
                reasoning: ReasoningSupport {
                    efforts: vec![Low, Medium, High, XHigh, Max],
                    budget: true,
                    disable: true,
                    output: true,
                },
                native_compaction: true,
                seed: false,
                presence_penalty: false,
                frequency_penalty: false,
                ..full
            },
            ApiProfile::GeminiGenerateContent => Self {
                reasoning: ReasoningSupport {
                    efforts: vec![Minimal, Low, Medium, High],
                    budget: true,
                    disable: true,
                    output: true,
                },
                strict_tools: false,
                parallel_tool_calls: false,
                ..full
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn support(efforts: &[ReasoningEffort], budget: bool) -> ReasoningSupport {
        ReasoningSupport {
            efforts: efforts.to_vec(),
            budget,
            disable: true,
            output: true,
        }
    }

    #[test]
    fn nearest_effort_prefers_the_cheaper_neighbour() {
        let anthropic = support(
            &[
                ReasoningEffort::Low,
                ReasoningEffort::Medium,
                ReasoningEffort::High,
                ReasoningEffort::Max,
            ],
            true,
        );
        assert_eq!(
            anthropic.nearest_effort(ReasoningEffort::Minimal),
            Some(ReasoningEffort::Low)
        );
        assert_eq!(
            anthropic.nearest_effort(ReasoningEffort::XHigh),
            Some(ReasoningEffort::High)
        );
        assert_eq!(
            anthropic.nearest_effort(ReasoningEffort::Max),
            Some(ReasoningEffort::Max)
        );
        assert_eq!(
            support(&[], true).nearest_effort(ReasoningEffort::Low),
            None
        );
    }

    #[test]
    fn budgets_bucket_into_efforts() {
        let openai = support(
            &[
                ReasoningEffort::Minimal,
                ReasoningEffort::Low,
                ReasoningEffort::Medium,
                ReasoningEffort::High,
            ],
            false,
        );
        let budget = |tokens| NonZeroU32::new(tokens).unwrap();
        assert_eq!(
            openai.effort_for_budget(budget(512)),
            Some(ReasoningEffort::Minimal)
        );
        assert_eq!(
            openai.effort_for_budget(budget(4096)),
            Some(ReasoningEffort::Low)
        );
        assert_eq!(
            openai.effort_for_budget(budget(16384)),
            Some(ReasoningEffort::Medium)
        );
        assert_eq!(
            openai.effort_for_budget(budget(100_000)),
            Some(ReasoningEffort::High)
        );
    }
}
