use std::sync::Arc;

use llmwire::transport::mock::MockTransport;
use llmwire::{Client, Credentials, Provider, ProviderConfig};

pub(crate) fn client_with(mock: &Arc<MockTransport>) -> Client {
    Client::builder()
        .http_transport(mock.clone() as Arc<dyn llmwire::transport::HttpTransport>)
        .build()
        .expect("client builds")
}

pub(crate) fn provider_with(mock: &Arc<MockTransport>, config: ProviderConfig) -> Provider {
    client_with(mock).provider(config).expect("provider builds")
}

pub(crate) fn openai_responses(mock: &Arc<MockTransport>) -> Provider {
    provider_with(
        mock,
        ProviderConfig::openai_responses(Credentials::api_key("sk-test")),
    )
}

pub(crate) fn openai_chat(mock: &Arc<MockTransport>) -> Provider {
    provider_with(
        mock,
        ProviderConfig::openai_chat(Credentials::api_key("sk-test")),
    )
}

pub(crate) fn openrouter(mock: &Arc<MockTransport>) -> Provider {
    provider_with(
        mock,
        ProviderConfig::openrouter(Credentials::api_key("sk-or-test")),
    )
}

pub(crate) fn chatgpt(mock: &Arc<MockTransport>) -> Provider {
    provider_with(
        mock,
        ProviderConfig::chatgpt(Credentials::bearer("chatgpt-access-token")),
    )
}

pub(crate) fn xai(mock: &Arc<MockTransport>) -> Provider {
    provider_with(mock, ProviderConfig::xai(Credentials::api_key("xai-key")))
}

pub(crate) fn xai_chat(mock: &Arc<MockTransport>) -> Provider {
    provider_with(
        mock,
        ProviderConfig::xai_chat(Credentials::api_key("xai-key")),
    )
}

pub(crate) fn anthropic(mock: &Arc<MockTransport>) -> Provider {
    provider_with(
        mock,
        ProviderConfig::anthropic(Credentials::api_key("sk-ant-test")),
    )
}

pub(crate) fn gemini(mock: &Arc<MockTransport>) -> Provider {
    provider_with(mock, ProviderConfig::gemini(Credentials::api_key("g-test")))
}
