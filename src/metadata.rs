//! Namespaced provider data attached to requests, results, messages, and parts.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Provider-specific JSON grouped by API-profile namespace.
///
/// The library reserves `"cogfer"` for its own fields. Custom metadata
/// should use another namespace.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProviderMetadata(BTreeMap<String, Value>);

impl ProviderMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(namespace: impl Into<String>, value: Value) -> Self {
        let mut map = BTreeMap::new();
        map.insert(namespace.into(), value);
        Self(map)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, namespace: &str) -> Option<&Value> {
        self.0.get(namespace)
    }

    /// Namespaces and their values in namespace order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.0
            .iter()
            .map(|(namespace, value)| (namespace.as_str(), value))
    }

    pub fn into_inner(self) -> BTreeMap<String, Value> {
        self.0
    }

    pub fn insert(&mut self, namespace: impl Into<String>, value: Value) {
        self.0.insert(namespace.into(), value);
    }

    /// Deep-merge `other`, using null values to remove keys.
    pub fn merge(&mut self, other: ProviderMetadata) {
        for (namespace, value) in other.0 {
            match self.0.get_mut(&namespace) {
                Some(existing) => {
                    if value.is_null() {
                        self.0.remove(&namespace);
                    } else {
                        crate::util::json_merge(existing, value);
                    }
                }
                None => {
                    if !value.is_null() {
                        self.0.insert(namespace, value);
                    }
                }
            }
        }
    }
}

impl From<BTreeMap<String, Value>> for ProviderMetadata {
    fn from(map: BTreeMap<String, Value>) -> Self {
        Self(map)
    }
}
