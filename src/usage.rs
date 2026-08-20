//! Normalized token usage.

/// Token usage for a completion.
///
/// Every field is optional: `None` means the provider did not report the
/// value, which is distinct from an explicit `0`.
///
/// `input_tokens` counts fresh input after subtracting provider cache reads
/// and writes. [`Usage::total_input_tokens`] returns the full prompt size.
/// Multi-pass turns report the summed cost of every pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    /// Tokens read from a provider prompt cache.
    pub cached_input_tokens: Option<u64>,
    /// Tokens written to a provider prompt cache (Anthropic).
    pub cache_creation_input_tokens: Option<u64>,
    /// Reasoning/thinking tokens included in `output_tokens`.
    pub reasoning_tokens: Option<u64>,
    /// Tokens added to the prompt for tool definitions (Gemini).
    pub tool_use_prompt_tokens: Option<u64>,
}

impl Usage {
    /// Whether the provider reported anything at all.
    pub fn is_empty(&self) -> bool {
        self == &Usage::default()
    }

    /// Combined prompt-side tokens: input + cache reads + cache writes,
    /// treating missing values as zero. Returns `None` if nothing was reported.
    pub fn total_input_tokens(&self) -> Option<u64> {
        if self.input_tokens.is_none()
            && self.cached_input_tokens.is_none()
            && self.cache_creation_input_tokens.is_none()
        {
            return None;
        }
        Some(
            self.input_tokens
                .unwrap_or(0)
                .saturating_add(self.cached_input_tokens.unwrap_or(0))
                .saturating_add(self.cache_creation_input_tokens.unwrap_or(0)),
        )
    }

    /// Merge a later usage report into this one.
    ///
    /// Later `Some` values win because streaming usage is cumulative.
    pub fn merge_from(&mut self, later: &Usage) {
        macro_rules! take_later {
            ($field:ident) => {
                if later.$field.is_some() {
                    self.$field = later.$field;
                }
            };
        }
        take_later!(input_tokens);
        take_later!(output_tokens);
        take_later!(total_tokens);
        take_later!(cached_input_tokens);
        take_later!(cache_creation_input_tokens);
        take_later!(reasoning_tokens);
        take_later!(tool_use_prompt_tokens);
    }

    /// Add another usage report to this one, field by field.
    ///
    /// Sum multi-pass usage field by field. `None + None` stays `None`.
    pub fn add_from(&mut self, other: &Usage) {
        macro_rules! add {
            ($field:ident) => {
                self.$field = match (self.$field, other.$field) {
                    (None, None) => None,
                    (a, b) => Some(a.unwrap_or(0).saturating_add(b.unwrap_or(0))),
                };
            };
        }
        add!(input_tokens);
        add!(output_tokens);
        add!(total_tokens);
        add!(cached_input_tokens);
        add!(cache_creation_input_tokens);
        add!(reasoning_tokens);
        add!(tool_use_prompt_tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_input_tokens_saturates_provider_counters() {
        let usage = Usage {
            input_tokens: Some(u64::MAX),
            cached_input_tokens: Some(1),
            ..Usage::default()
        };

        assert_eq!(usage.total_input_tokens(), Some(u64::MAX));
    }

    #[test]
    fn add_from_saturates_provider_counters() {
        let mut usage = Usage {
            output_tokens: Some(u64::MAX),
            ..Usage::default()
        };
        usage.add_from(&Usage {
            output_tokens: Some(1),
            ..Usage::default()
        });

        assert_eq!(usage.output_tokens, Some(u64::MAX));
    }
}
