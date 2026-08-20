use crate::usage::Usage;

/// Normalize provider token accounting to SDK-wide semantics.
pub(crate) fn normalize_usage(
    input: Option<u64>,
    output: Option<u64>,
    total: Option<u64>,
    cached: Option<u64>,
    cache_write: Option<u64>,
    reasoning: Option<u64>,
) -> Usage {
    Usage {
        input_tokens: input.map(|input| {
            input.saturating_sub(cached.unwrap_or(0).saturating_add(cache_write.unwrap_or(0)))
        }),
        output_tokens: output,
        total_tokens: total,
        cached_input_tokens: cached,
        cache_creation_input_tokens: cache_write,
        reasoning_tokens: reasoning,
        tool_use_prompt_tokens: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_usage_saturates_combined_cache_counters() {
        let usage = normalize_usage(Some(10), None, None, Some(u64::MAX), Some(1), None);

        assert_eq!(usage.input_tokens, Some(0));
    }

    #[test]
    fn normalize_usage_preserves_remainder_at_u64_limit() {
        let usage = normalize_usage(Some(u64::MAX), None, None, Some(u64::MAX - 1), None, None);

        assert_eq!(usage.input_tokens, Some(1));
    }
}
