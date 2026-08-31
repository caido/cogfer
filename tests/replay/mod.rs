//! Replay tests: every API profile runs the shared [`scenarios`] against
//! cassettes recorded from the real provider (`tests/cassettes/<profile>/`).
//!
//! Replay needs no network or credentials and runs in CI. The same scenario
//! function runs in both modes, so its assertions were verified against the
//! live provider when the cassette was recorded and are checked again on every
//! replay against the exact bytes that provider sent.
//!
//! Re-record a profile when the provider's wire format changes or a scenario is added:
//!
//! ```sh
//! cargo test --test lib replay::openai_responses::record -- --ignored --nocapture
//! ```
//!
//! Recording uses the credentials the live tests use (`OPENAI_API_KEY`,
//! `GOOGLE_API_KEY`, ..., or the cached ChatGPT/SpaceXAI sign-ins), runs each
//! scenario against the live provider, and writes the exchanges it saw. A
//! scenario that fails live is not recorded and is reported at the end. Set
//! `RECORD_SCENARIOS=stream_text,unknown_model` to re-record only those and
//! leave the other cassettes untouched. Review a fresh cassette before
//! committing it: request header values outside a small allowlist are
//! redacted, but bodies are stored as sent.

#[cfg(feature = "reqwest-transport")]
mod record;
pub(crate) mod scenarios;

mod anthropic;
#[cfg(feature = "aws")]
mod bedrock_anthropic;
mod chatgpt;
mod gemini;
mod openai_chat;
mod openai_responses;
mod openrouter;
mod xai_chat;
mod xai_responses;

use std::sync::Arc;

use llmwire::transport::mock::MockTransport;
use llmwire::{Credentials, Provider, ProviderConfig};

#[cfg(feature = "reqwest-transport")]
pub(crate) use self::record::RecordSession;
use crate::common::{Body, Cassette, provider_with, scrub_json};

/// Declare a replay suite: one test per scenario plus an ignored `record`
/// test that re-records all of them.
///
/// - `config`: the [`ProviderConfig`] constructor for the profile.
/// - `live`: `fn(&Client) -> Option<Provider>` building the recording provider
///   from real credentials, or `None` to skip.
/// - `model`: the model to record against. Replay uses the model stored in
///   each cassette.
macro_rules! cassette_suite {
    (
        config: $config:expr,
        live: $live:expr,
        model: $model:expr,
        scenarios: [$($scenario:ident),+ $(,)?] $(,)?
    ) => {
        $(
            #[tokio::test]
            async fn $scenario() {
                let replay = $crate::replay::Replay::start($config, stringify!($scenario));
                $crate::replay::scenarios::$scenario(&replay.provider, &replay.model).await;
                replay.finish();
            }
        )+

        #[cfg(feature = "reqwest-transport")]
        #[tokio::test]
        #[ignore = "records cassettes against the live provider"]
        async fn record() {
            let Some(session) = $crate::replay::RecordSession::start($live, $model) else {
                return;
            };
            let mut failed = Vec::new();
            $(
                let scenario = stringify!($scenario);
                if session.selected(scenario) {
                    let run = $crate::replay::scenarios::$scenario(&session.provider, session.model);
                    if !session.record(scenario, run).await {
                        failed.push(scenario);
                    }
                }
            )+
            assert!(
                failed.is_empty(),
                "scenarios failed against the live provider and were not recorded: {failed:?}"
            );
        }
    };
}
pub(crate) use cassette_suite;

/// One scenario replayed from its cassette.
pub(crate) struct Replay {
    pub(crate) provider: Provider,
    /// The model the cassette was recorded with.
    pub(crate) model: String,
    mock: Arc<MockTransport>,
    cassette: Cassette,
}

impl Replay {
    /// Load the cassette for `scenario` and queue it on a fresh mock transport.
    pub(crate) fn start(config: fn(Credentials) -> ProviderConfig, scenario: &str) -> Self {
        let config = config(Credentials::api_key("cassette"));
        let profile = config.profile();
        let cassette = Cassette::load(profile, scenario).unwrap_or_else(|| {
            panic!(
                "no cassette at {}; record it with \
                 `cargo test --test lib replay::{}::record -- --ignored`",
                Cassette::path(profile, scenario).display(),
                profile.as_str().replace('-', "_"),
            )
        });
        assert_eq!(
            cassette.profile,
            profile.as_str(),
            "cassette was recorded for another profile"
        );
        let mock = MockTransport::shared();
        cassette.queue(&mock);
        Self {
            provider: provider_with(&mock, config),
            model: cassette.model.clone(),
            mock,
            cassette,
        }
    }

    /// Check that the scenario sent exactly the recorded requests: same
    /// count, URLs, and JSON bodies. Request lowering is deterministic given
    /// the cassette's responses, so this doubles as a lowering regression
    /// test for every profile. An intentional lowering change means updating
    /// the recorded request (or re-recording).
    pub(crate) fn finish(self) {
        let requests = self.mock.requests();
        assert_eq!(
            requests.len(),
            self.cassette.exchanges.len(),
            "scenario made a different number of requests than the cassette holds"
        );
        for (index, (sent, exchange)) in requests.iter().zip(&self.cassette.exchanges).enumerate() {
            let recorded = &exchange.request;
            assert_eq!(
                sent.url.as_str(),
                recorded.url,
                "request {index} URL differs"
            );
            if let Some(Body::Json(recorded_body)) = &recorded.body {
                let mut sent_body: serde_json::Value = serde_json::from_slice(
                    sent.body.as_ref().expect("recorded requests have bodies"),
                )
                .expect("request body is JSON");
                scrub_json(&mut sent_body);
                assert!(
                    &sent_body == recorded_body,
                    "request {index} body differs from the cassette\n--- sent ---\n{}\n--- recorded ---\n{}",
                    serde_json::to_string_pretty(&sent_body).unwrap(),
                    serde_json::to_string_pretty(recorded_body).unwrap(),
                );
            }
        }
    }
}
