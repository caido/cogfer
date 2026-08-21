//! OpenAI Responses API and the backends that speak its wire format.

use crate::error::{Error, Result};
use crate::http::join_url;
use crate::protocols::openai::shared::decode_openai_error;
use crate::protocols::{
    ApiProfile, LoweredRequest, ProtocolContext, ProtocolHandler, StreamDecoder,
};
use crate::response::GenerateResult;
use crate::transport::{HttpRequest, HttpResponse};

pub(crate) mod chatgpt;
mod request;
mod stream;
mod types;

use self::request::lower_body;
use self::stream::ResponsesStreamDecoder;
use self::types::decode_openai_response;

/// Which backend speaks the Responses wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResponsesDialect {
    /// The official OpenAI API.
    OpenAi,
    /// xAI's Responses-compatible API.
    Xai,
    /// The ChatGPT subscription backend.
    ChatGpt,
}

impl ResponsesDialect {
    pub(crate) fn profile(self) -> ApiProfile {
        match self {
            ResponsesDialect::OpenAi => ApiProfile::OpenAiResponses,
            ResponsesDialect::Xai => ApiProfile::XaiResponses,
            ResponsesDialect::ChatGpt => ApiProfile::ChatGptResponses,
        }
    }

    pub(super) fn is_chatgpt(self) -> bool {
        matches!(self, Self::ChatGpt)
    }
}

pub(crate) struct Handler {
    dialect: ResponsesDialect,
}

impl Handler {
    pub(crate) const OPENAI: Self = Self {
        dialect: ResponsesDialect::OpenAi,
    };
    pub(crate) const XAI: Self = Self {
        dialect: ResponsesDialect::Xai,
    };
}

impl ProtocolHandler for Handler {
    fn lower(&self, ctx: &ProtocolContext<'_>, streaming: bool) -> Result<LoweredRequest> {
        let (body, warnings) = lower_body(ctx, streaming, self.dialect)?;
        Ok(LoweredRequest {
            http: HttpRequest::post_json(join_url(ctx.base_url, "responses"), &body)?,
            warnings,
        })
    }

    fn decode_response(
        &self,
        ctx: &ProtocolContext<'_>,
        response: &HttpResponse,
    ) -> Result<GenerateResult> {
        decode_openai_response(ctx, response)
    }

    fn decode_error(&self, status: u16, headers: &[(String, String)], body: &[u8]) -> Error {
        decode_openai_error(self.dialect.profile(), status, headers, body)
    }

    fn new_stream_decoder(&self, _ctx: &ProtocolContext<'_>) -> Box<dyn StreamDecoder> {
        Box::new(ResponsesStreamDecoder::new(self.dialect.profile()))
    }
}
