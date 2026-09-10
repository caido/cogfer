//! OpenAI Responses API and the backends that speak its wire format.

use url::Url;

use crate::error::{Error, Result};
use crate::http::join_url;
use crate::protocols::openai::shared::decode_openai_error;
use crate::protocols::{
    ApiProfile, LoweredRequest, ProtocolContext, ProtocolHandler, StreamDecoder,
};
use crate::response::GenerateResult;
use crate::transport::{HeaderMap, HttpRequest, HttpResponse};

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
    /// SpaceXAI's Responses-compatible API.
    Xai,
    /// The ChatGPT subscription backend.
    ChatGpt,
    /// Amazon Bedrock's OpenAI-compatible endpoint.
    #[cfg(feature = "aws")]
    Bedrock,
}

impl ResponsesDialect {
    pub(crate) fn profile(self) -> ApiProfile {
        match self {
            ResponsesDialect::OpenAi => ApiProfile::OpenAiResponses,
            ResponsesDialect::Xai => ApiProfile::XaiResponses,
            ResponsesDialect::ChatGpt => ApiProfile::ChatGptResponses,
            #[cfg(feature = "aws")]
            ResponsesDialect::Bedrock => ApiProfile::BedrockOpenAiResponses,
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
    #[cfg(feature = "aws")]
    pub(crate) const BEDROCK: Self = Self {
        dialect: ResponsesDialect::Bedrock,
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

    fn new_verify_request(&self, base_url: &Url) -> Result<HttpRequest> {
        #[cfg(feature = "aws")]
        if self.dialect == ResponsesDialect::Bedrock {
            return Ok(crate::protocols::bedrock::new_verify_request(base_url));
        }
        Ok(HttpRequest::get(join_url(base_url, "models")))
    }

    fn decode_response(
        &self,
        ctx: &ProtocolContext<'_>,
        response: &HttpResponse,
    ) -> Result<GenerateResult> {
        decode_openai_response(ctx, response)
    }

    fn decode_error(&self, status: u16, headers: &HeaderMap, body: &[u8]) -> Error {
        // Bedrock answers endpoint-level failures (auth, throttling,
        // validation) with an AWS envelope and model-level failures with the
        // OpenAI one; the exception-class header tells them apart.
        #[cfg(feature = "aws")]
        if self.dialect == ResponsesDialect::Bedrock
            && headers.contains_key(crate::protocols::bedrock::ERROR_TYPE)
        {
            return crate::protocols::bedrock::decode_bedrock_error(
                ApiProfile::BedrockOpenAiResponses,
                status,
                headers,
                body,
            );
        }
        decode_openai_error(self.dialect.profile(), status, headers, body)
    }

    fn new_stream_decoder(&self, _ctx: &ProtocolContext<'_>) -> Box<dyn StreamDecoder> {
        Box::new(ResponsesStreamDecoder::new(self.dialect.profile()))
    }
}
