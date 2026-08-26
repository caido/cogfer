//! OpenAI Chat Completions and the dialects built on it.

use url::Url;

use crate::error::{Error, Result};
use crate::protocols::openai::shared::decode_openai_error;
use crate::protocols::{
    ApiProfile, LoweredRequest, ProtocolContext, ProtocolHandler, StreamDecoder,
};
use crate::response::GenerateResult;
use crate::transport::{HeaderMap, HttpResponse};

mod request;
mod stream;
mod types;
use self::request::lower_chat;
use self::stream::ChatStreamDecoder;
use self::types::{decode_chat_response, decode_openrouter_error};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputTokenAccounting {
    /// OpenAI and OpenRouter include reasoning in completion tokens.
    IncludesReasoning,
    /// xAI reports completion tokens without reasoning, so normalization adds it.
    ExcludesReasoning,
}

/// Which flavor of the Chat Completions wire format to speak.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChatDialect {
    /// OpenAI itself (and Azure OpenAI): the full wire format.
    OpenAi,
    /// Other Chat Completions servers: the widely supported subset, without
    /// OpenAI-only spellings that strict servers reject as unknown fields.
    Compatible,
    Xai,
    OpenRouter,
}

impl ChatDialect {
    pub(crate) fn profile(self) -> ApiProfile {
        match self {
            ChatDialect::OpenAi | ChatDialect::Compatible => ApiProfile::OpenAiChatCompletions,
            ChatDialect::Xai => ApiProfile::XaiChatCompletions,
            ChatDialect::OpenRouter => ApiProfile::OpenRouter,
        }
    }

    pub(crate) fn output_token_accounting(self) -> OutputTokenAccounting {
        match self {
            ChatDialect::Xai => OutputTokenAccounting::ExcludesReasoning,
            ChatDialect::OpenAi | ChatDialect::Compatible | ChatDialect::OpenRouter => {
                OutputTokenAccounting::IncludesReasoning
            }
        }
    }

    /// The dialect to speak to `base_url`: the OpenAI profile only uses the
    /// full wire format against OpenAI's own endpoints.
    fn for_endpoint(self, base_url: &Url) -> Self {
        if self == ChatDialect::OpenAi && !is_openai_endpoint(base_url) {
            ChatDialect::Compatible
        } else {
            self
        }
    }
}

fn is_openai_endpoint(base_url: &Url) -> bool {
    let Some(host) = base_url.host_str() else {
        return false;
    };
    host.eq_ignore_ascii_case("api.openai.com")
        || [
            ".openai.azure.com",
            ".services.ai.azure.com",
            ".cognitiveservices.azure.com",
        ]
        .iter()
        .any(|suffix| host.to_ascii_lowercase().ends_with(suffix))
}

pub(crate) struct Handler {
    dialect: ChatDialect,
}

impl Handler {
    pub(crate) const OPENAI: Self = Self {
        dialect: ChatDialect::OpenAi,
    };
    pub(crate) const XAI: Self = Self {
        dialect: ChatDialect::Xai,
    };
    pub(crate) const OPENROUTER: Self = Self {
        dialect: ChatDialect::OpenRouter,
    };
}

impl ProtocolHandler for Handler {
    fn lower(&self, ctx: &ProtocolContext<'_>, streaming: bool) -> Result<LoweredRequest> {
        lower_chat(ctx, streaming, self.dialect.for_endpoint(ctx.base_url))
    }

    fn decode_response(
        &self,
        _ctx: &ProtocolContext<'_>,
        response: &HttpResponse,
    ) -> Result<GenerateResult> {
        decode_chat_response(response, self.dialect)
    }

    fn decode_error(&self, status: u16, headers: &HeaderMap, body: &[u8]) -> Error {
        match self.dialect {
            ChatDialect::OpenRouter => decode_openrouter_error(status, headers, body),
            _ => decode_openai_error(self.dialect.profile(), status, headers, body),
        }
    }

    fn new_stream_decoder(&self, _ctx: &ProtocolContext<'_>) -> Box<dyn StreamDecoder> {
        Box::new(ChatStreamDecoder::new(self.dialect))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_openai_endpoints_get_the_full_wire_format() {
        for (url, expected) in [
            ("https://api.openai.com/v1", ChatDialect::OpenAi),
            (
                "https://res.openai.azure.com/openai/v1",
                ChatDialect::OpenAi,
            ),
            ("http://localhost:11434/v1", ChatDialect::Compatible),
            ("https://api.mistral.ai/v1", ChatDialect::Compatible),
        ] {
            let url = Url::parse(url).unwrap();
            assert_eq!(ChatDialect::OpenAi.for_endpoint(&url), expected, "{url}");
            assert_eq!(ChatDialect::Xai.for_endpoint(&url), ChatDialect::Xai);
        }
    }
}
