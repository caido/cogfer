use std::sync::Mutex;

use llmwire::{Error, ErrorKind, TokenStore};

pub(crate) fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time should be after the Unix epoch")
        .as_secs() as i64
}

pub(crate) struct RecordingTokenStore<T> {
    saved: Mutex<Option<T>>,
    fail: bool,
}

impl<T> Default for RecordingTokenStore<T> {
    fn default() -> Self {
        Self {
            saved: Mutex::new(None),
            fail: false,
        }
    }
}

impl<T: Clone> RecordingTokenStore<T> {
    pub(crate) fn failing() -> Self {
        Self {
            saved: Mutex::new(None),
            fail: true,
        }
    }

    pub(crate) fn saved(&self) -> Option<T> {
        self.saved.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl<T: Clone + Send + Sync> TokenStore<T> for RecordingTokenStore<T> {
    async fn save(&self, tokens: &T) -> llmwire::Result<()> {
        if self.fail {
            return Err(Error::new(ErrorKind::Provider, "token store failed"));
        }
        *self.saved.lock().unwrap() = Some(tokens.clone());
        Ok(())
    }
}
