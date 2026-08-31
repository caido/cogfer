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

/// Decode base64 in the standard or URL-safe alphabet, tolerating optional
/// padding.
pub(crate) fn base64_decode(input: &str, url_safe: bool) -> Option<Vec<u8>> {
    let value = |byte: u8| -> Option<u32> {
        match byte {
            b'A'..=b'Z' => Some(u32::from(byte - b'A')),
            b'a'..=b'z' => Some(u32::from(byte - b'a') + 26),
            b'0'..=b'9' => Some(u32::from(byte - b'0') + 52),
            b'-' if url_safe => Some(62),
            b'_' if url_safe => Some(63),
            b'+' if !url_safe => Some(62),
            b'/' if !url_safe => Some(63),
            _ => None,
        }
    };

    let input = input.trim_end_matches('=').as_bytes();
    let mut output = Vec::with_capacity(input.len() * 3 / 4);
    for chunk in input.chunks(4) {
        if chunk.len() == 1 {
            return None;
        }
        let mut buffer: u32 = 0;
        for byte in chunk {
            buffer = (buffer << 6) | value(*byte)?;
        }
        buffer <<= 6 * (4 - chunk.len()) as u32;
        let bytes = buffer.to_be_bytes();
        output.extend_from_slice(&bytes[1..chunk.len()]);
    }
    Some(output)
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

    #[test]
    fn base64_decodes_both_alphabets() {
        assert_eq!(base64_decode("+/8=", false).unwrap(), [0xFB, 0xFF]);
        assert_eq!(base64_decode("-_8", true).unwrap(), [0xFB, 0xFF]);
        assert_eq!(base64_decode("+/8=", true), None);
        assert_eq!(base64_decode("-_8", false), None);
        assert_eq!(base64_decode("aGk=", false).unwrap(), b"hi");
        assert_eq!(base64_decode("a", false), None);
    }
}
