//! Recording driver for the replay suites: a live provider on a
//! [`RecordingTransport`] plus per-scenario save/discard.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use futures_util::FutureExt;
use llmwire::transport::ReqwestTransport;
use llmwire::{Client, Provider};

use crate::common::RecordingTransport;

/// A live provider whose transport records every exchange.
pub(crate) struct RecordSession {
    pub(crate) provider: Provider,
    /// The model every scenario in this session targets.
    pub(crate) model: &'static str,
    recorder: Arc<RecordingTransport>,
}

impl RecordSession {
    /// Build the recording provider, or `None` when `live` has no credentials.
    pub(crate) fn start(
        live: impl FnOnce(&Client) -> Option<Provider>,
        model: &'static str,
    ) -> Option<Self> {
        llmwire::transport::install_default_crypto_provider();
        let recorder = Arc::new(RecordingTransport::new(Arc::new(
            ReqwestTransport::new().expect("reqwest transport builds"),
        )));
        let client = Client::builder()
            .http_transport(recorder.clone())
            .build()
            .expect("client builds");
        let provider = live(&client)?;
        Some(Self {
            provider,
            model,
            recorder,
        })
    }

    /// Whether `scenario` should be recorded: all are unless `RECORD_SCENARIOS`
    /// names a comma-separated subset.
    pub(crate) fn selected(&self, scenario: &str) -> bool {
        match std::env::var("RECORD_SCENARIOS") {
            Ok(selected) => selected.split(',').any(|name| name.trim() == scenario),
            Err(_) => true,
        }
    }

    /// Run `scenario` against the live provider and save its exchanges.
    /// Returns whether it passed. A failing scenario is not recorded.
    pub(crate) async fn record(&self, scenario: &str, run: impl Future<Output = ()>) -> bool {
        match AssertUnwindSafe(run).catch_unwind().await {
            Ok(()) => {
                self.recorder
                    .save(self.provider.profile(), self.model, scenario);
                true
            }
            Err(_) => {
                self.recorder.discard();
                eprintln!("scenario `{scenario}` failed against the live provider");
                false
            }
        }
    }
}
