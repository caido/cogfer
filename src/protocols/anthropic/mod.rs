//! Anthropic Messages API (`POST {base}/messages`) and the backends that
//! serve it.

use super::{ApiProfile, LoweredRequest, ProtocolContext, ProtocolHandler, StreamDecoder};
use crate::error::{Error, Result};
use crate::response::GenerateResult;
use crate::transport::aws_event_stream::AwsEventStreamParser;
use crate::transport::framing::FrameSource;
use crate::transport::{HeaderMap, HeaderName, HttpRequest, HttpResponse};

mod bedrock;
mod request;
mod stream;
mod types;

use self::bedrock::{BedrockStreamDecoder, decode_bedrock_error};
use self::request::lower_anthropic_request;
use self::stream::AnthropicStreamDecoder;
use self::types::{decode_anthropic_error, decode_anthropic_response};

/// Which backend serves the Messages API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnthropicDialect {
    /// Anthropic's own API.
    Direct,
    /// Amazon Bedrock.
    Bedrock,
}

pub(crate) struct Handler {
    dialect: AnthropicDialect,
}

impl Handler {
    pub(crate) const DIRECT: Self = Self {
        dialect: AnthropicDialect::Direct,
    };
    pub(crate) const BEDROCK: Self = Self {
        dialect: AnthropicDialect::Bedrock,
    };
}

impl ProtocolHandler for Handler {
    fn lower(&self, ctx: &ProtocolContext<'_>, streaming: bool) -> Result<LoweredRequest> {
        lower_anthropic_request(ctx, streaming, self.dialect)
    }

    fn decode_response(
        &self,
        _ctx: &ProtocolContext<'_>,
        response: &HttpResponse,
    ) -> Result<GenerateResult> {
        decode_anthropic_response(response)
    }

    fn decode_error(&self, status: u16, headers: &HeaderMap, body: &[u8]) -> Error {
        match self.dialect {
            AnthropicDialect::Direct => decode_anthropic_error(status, headers, body),
            AnthropicDialect::Bedrock => decode_bedrock_error(status, headers, body),
        }
    }

    fn new_stream_decoder(&self, _ctx: &ProtocolContext<'_>) -> Box<dyn StreamDecoder> {
        match self.dialect {
            AnthropicDialect::Direct => {
                Box::new(AnthropicStreamDecoder::new(ApiProfile::AnthropicMessages))
            }
            AnthropicDialect::Bedrock => Box::new(BedrockStreamDecoder::new()),
        }
    }

    fn new_frame_source(&self) -> Box<dyn FrameSource> {
        match self.dialect {
            AnthropicDialect::Direct => Box::new(crate::transport::sse::SseParser::new()),
            AnthropicDialect::Bedrock => Box::new(AwsEventStreamParser::new()),
        }
    }

    /// Anthropic takes API keys in `x-api-key`. Bedrock API keys are bearer
    /// tokens, and SigV4 arrives through a
    /// [`RequestAuthenticator`](crate::RequestAuthenticator).
    fn apply_auth(
        &self,
        request: &mut HttpRequest,
        credentials: &crate::auth::Credentials,
    ) -> Result<()> {
        match self.dialect {
            AnthropicDialect::Direct => {
                credentials.apply_native_key(request, HeaderName::from_static("x-api-key"))
            }
            AnthropicDialect::Bedrock => credentials.apply_bearer(request),
        }
    }
}
