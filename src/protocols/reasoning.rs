//! Fit a [`ReasoningConfig`] to what a profile can express.
//!
//! Callers choose between a discrete effort and a token budget; providers
//! accept one or the other, and not every effort level. Rather than each
//! protocol dropping what it cannot send, this maps the request onto the
//! closest supported control and records the approximation as a warning.

use std::num::NonZeroU32;

use super::ApiProfile;
use crate::capabilities::ReasoningSupport;
use crate::request::{ReasoningConfig, ReasoningEffort, ReasoningOutput};
use crate::response::Warning;

/// A reasoning setting the profile can send verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResolvedReasoning {
    Disabled,
    Effort(ReasoningEffort),
    Budget(NonZeroU32),
}

/// The reasoning control to send, plus the output visibility when the
/// profile can honor it. `None` when nothing can be sent.
pub(crate) fn resolve(
    config: ReasoningConfig,
    support: &ReasoningSupport,
    profile: ApiProfile,
    warnings: &mut Vec<Warning>,
) -> Option<(ResolvedReasoning, Option<ReasoningOutput>)> {
    let (requested_output, resolved) = match config {
        ReasoningConfig::Disabled => {
            if !support.disable {
                warnings.push(Warning::unsupported_setting(
                    "reasoning",
                    format!("{profile} cannot disable reasoning"),
                ));
                return None;
            }
            (None, ResolvedReasoning::Disabled)
        }
        ReasoningConfig::Effort { effort, output } => {
            (output, resolve_effort(effort, support, profile, warnings)?)
        }
        ReasoningConfig::Budget { tokens, output } => {
            (output, resolve_budget(tokens, support, profile, warnings)?)
        }
    };
    let output = match requested_output {
        Some(_) if !support.output => {
            warnings.push(Warning::unsupported_setting(
                "reasoning.output",
                format!("{profile} cannot control reasoning output visibility"),
            ));
            None
        }
        output => output,
    };
    Some((resolved, output))
}

fn resolve_effort(
    effort: ReasoningEffort,
    support: &ReasoningSupport,
    profile: ApiProfile,
    warnings: &mut Vec<Warning>,
) -> Option<ResolvedReasoning> {
    if let Some(nearest) = support.nearest_effort(effort) {
        if nearest != effort {
            warnings.push(Warning::approximated_setting(
                "reasoning.effort",
                format!(
                    "{profile} does not accept reasoning effort `{}`; sent `{}`",
                    effort.as_str(),
                    nearest.as_str()
                ),
            ));
        }
        return Some(ResolvedReasoning::Effort(nearest));
    }
    if support.budget {
        let tokens = ReasoningSupport::budget_for_effort(effort);
        warnings.push(Warning::approximated_setting(
            "reasoning.effort",
            format!(
                "{profile} uses reasoning token budgets; effort `{}` was sent as {tokens} tokens",
                effort.as_str()
            ),
        ));
        return Some(ResolvedReasoning::Budget(tokens));
    }
    warnings.push(Warning::unsupported_setting(
        "reasoning.effort",
        format!("{profile} has no reasoning control"),
    ));
    None
}

fn resolve_budget(
    tokens: NonZeroU32,
    support: &ReasoningSupport,
    profile: ApiProfile,
    warnings: &mut Vec<Warning>,
) -> Option<ResolvedReasoning> {
    if support.budget {
        return Some(ResolvedReasoning::Budget(tokens));
    }
    if let Some(effort) = support.effort_for_budget(tokens) {
        warnings.push(Warning::approximated_setting(
            "reasoning.budget",
            format!(
                "{profile} uses discrete reasoning efforts; a budget of {tokens} tokens was sent as `{}`",
                effort.as_str()
            ),
        ));
        return Some(ResolvedReasoning::Effort(effort));
    }
    warnings.push(Warning::unsupported_setting(
        "reasoning.budget",
        format!("{profile} has no reasoning control"),
    ));
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::response::WarningKind;

    fn budget(tokens: u32) -> NonZeroU32 {
        NonZeroU32::new(tokens).unwrap()
    }

    #[test]
    fn budgets_become_efforts_with_an_approximation_warning() {
        let support = ReasoningSupport {
            efforts: vec![ReasoningEffort::Low, ReasoningEffort::High],
            budget: false,
            disable: true,
            output: false,
        };
        let mut warnings = Vec::new();

        let resolved = resolve(
            ReasoningConfig::budget(budget(40_000)),
            &support,
            ApiProfile::XaiChatCompletions,
            &mut warnings,
        );

        assert_eq!(
            resolved,
            Some((ResolvedReasoning::Effort(ReasoningEffort::High), None))
        );
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].kind, WarningKind::ApproximatedSetting);
        assert_eq!(warnings[0].subject.as_deref(), Some("reasoning.budget"));
    }

    #[test]
    fn efforts_become_budgets_when_only_budgets_exist() {
        let support = ReasoningSupport {
            efforts: Vec::new(),
            budget: true,
            disable: true,
            output: true,
        };
        let mut warnings = Vec::new();

        let resolved = resolve(
            ReasoningConfig::Effort {
                effort: ReasoningEffort::Medium,
                output: Some(ReasoningOutput::Omit),
            },
            &support,
            ApiProfile::GeminiGenerateContent,
            &mut warnings,
        );

        assert_eq!(
            resolved,
            Some((
                ResolvedReasoning::Budget(budget(16384)),
                Some(ReasoningOutput::Omit)
            ))
        );
        assert_eq!(warnings[0].kind, WarningKind::ApproximatedSetting);
    }

    #[test]
    fn unsupported_output_control_is_dropped_with_a_warning() {
        let support = ReasoningSupport {
            efforts: vec![ReasoningEffort::Medium],
            budget: false,
            disable: true,
            output: false,
        };
        let mut warnings = Vec::new();

        let resolved = resolve(
            ReasoningConfig::Effort {
                effort: ReasoningEffort::Medium,
                output: Some(ReasoningOutput::Include),
            },
            &support,
            ApiProfile::OpenAiChatCompletions,
            &mut warnings,
        );

        assert_eq!(
            resolved,
            Some((ResolvedReasoning::Effort(ReasoningEffort::Medium), None))
        );
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].kind, WarningKind::UnsupportedSetting);
    }

    #[test]
    fn nothing_is_sent_without_any_reasoning_control() {
        let mut warnings = Vec::new();

        let resolved = resolve(
            ReasoningConfig::effort(ReasoningEffort::High),
            &ReasoningSupport::none(),
            ApiProfile::OpenAiChatCompletions,
            &mut warnings,
        );

        assert_eq!(resolved, None);
        assert_eq!(warnings[0].kind, WarningKind::UnsupportedSetting);
    }
}
