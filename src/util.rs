//! Small internal helpers shared across modules.

use serde_json::Value;

/// Deep-merge `incoming`, using null values to remove keys.
pub(crate) fn json_merge(target: &mut Value, incoming: Value) {
    match (target, incoming) {
        (Value::Object(target_map), Value::Object(incoming_map)) => {
            for (key, value) in incoming_map {
                if value.is_null() {
                    target_map.remove(&key);
                } else if let Some(existing) = target_map.get_mut(&key) {
                    json_merge(existing, value);
                } else {
                    target_map.insert(key, value);
                }
            }
        }
        (target, incoming) => *target = incoming,
    }
}

/// Truncate a string for inclusion in error messages, keeping char boundaries.
pub(crate) fn truncate_for_error(input: &str, max: usize) -> String {
    if input.len() <= max {
        return input.to_string();
    }
    let mut end = max;
    while !input.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &input[..end])
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn merge_recurses_and_null_removes() {
        let mut target = json!({"a": {"b": 1, "c": 2}, "keep": true});
        json_merge(&mut target, json!({"a": {"b": 3, "c": null, "d": 4}}));
        assert_eq!(target, json!({"a": {"b": 3, "d": 4}, "keep": true}));
    }

    #[test]
    fn merge_replaces_non_objects() {
        let mut target = json!({"a": [1, 2]});
        json_merge(&mut target, json!({"a": [3]}));
        assert_eq!(target, json!({"a": [3]}));
    }

    #[test]
    fn truncation_preserves_utf8_boundaries() {
        assert_eq!(truncate_for_error("abéz", 3), "ab…");
    }
}
