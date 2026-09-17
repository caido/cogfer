use cogfer::{Credentials, Message, ProviderConfig, Request, StreamEvent};
use url::Url;

use crate::common::{assert_terminal_contract, drain, live_client};

#[tokio::test]
#[ignore = "live"]
async fn ollama_openai_compatible_endpoint() {
    let Ok(model_id) = std::env::var("OLLAMA_MODEL") else {
        eprintln!("SKIP: set OLLAMA_MODEL to a locally installed model");
        return;
    };
    let base = std::env::var("OLLAMA_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let provider = live_client()
        .provider(
            ProviderConfig::openai_chat(Credentials::none())
                .with_base_url(Url::parse(&base).unwrap()),
        )
        .expect("provider builds");

    eprintln!("ollama: using local model {model_id}");

    let request = Request::builder()
        .message(Message::user("Reply with one short word."))
        .max_output_tokens(60)
        .build();
    let result = provider
        .language_model(model_id.clone())
        .generate(request)
        .await
        .expect("ollama generate succeeds");
    assert!(!result.text().is_empty() || result.has_tool_calls());

    let request = Request::builder()
        .message(Message::user("Count from 1 to 5."))
        .max_output_tokens(80)
        .build();
    let events = drain(
        provider
            .language_model(model_id)
            .stream(request)
            .await
            .expect("ollama stream establishes"),
    )
    .await;
    assert_terminal_contract(&events);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta { .. }))
    );
}
