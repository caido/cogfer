//! Anthropic Messages API (`POST {base}/messages`) and the backends that
//! serve it.

use url::Url;

use super::{ApiProfile, LoweredRequest, ProtocolContext, ProtocolHandler, StreamDecoder};
use crate::error::{Error, Result};
use crate::http::join_url;
use crate::response::GenerateResult;
#[cfg(feature = "aws")]
use crate::transport::aws_event_stream::AwsEventStreamParser;
use crate::transport::framing::FrameSource;
use crate::transport::{HeaderMap, HeaderName, HeaderValue, HttpRequest, HttpResponse};

#[cfg(feature = "aws")]
pub(crate) mod bedrock;
mod request;
mod stream;
mod types;

#[cfg(feature = "aws")]
use self::bedrock::BedrockStreamDecoder;
use self::request::lower_anthropic_request;
use self::stream::AnthropicStreamDecoder;
use self::types::{decode_anthropic_error, decode_anthropic_response};

/// The header and value every request to Anthropic's own API declares.
const VERSION_HEADER: HeaderName = HeaderName::from_static("anthropic-version");
const VERSION: HeaderValue = HeaderValue::from_static("2023-06-01");

/// Which backend serves the Messages API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnthropicDialect {
    /// Anthropic's own API.
    Direct,
    /// Amazon Bedrock.
    #[cfg(feature = "aws")]
    Bedrock,
}

pub(crate) struct Handler {
    dialect: AnthropicDialect,
}

impl Handler {
    pub(crate) const DIRECT: Self = Self {
        dialect: AnthropicDialect::Direct,
    };
    #[cfg(feature = "aws")]
    pub(crate) const BEDROCK: Self = Self {
        dialect: AnthropicDialect::Bedrock,
    };
}

impl ProtocolHandler for Handler {
    fn lower(&self, ctx: &ProtocolContext<'_>, streaming: bool) -> Result<LoweredRequest> {
        lower_anthropic_request(ctx, streaming, self.dialect)
    }

    fn verify_request(&self, base_url: &Url) -> Result<HttpRequest> {
        match self.dialect {
            AnthropicDialect::Direct => {
                let mut http = HttpRequest::get(join_url(base_url, "models"));
                http.headers.insert(VERSION_HEADER, VERSION);
                Ok(http)
            }
            #[cfg(feature = "aws")]
            AnthropicDialect::Bedrock => Ok(crate::protocols::bedrock::verify_request(base_url)),
        }
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
            #[cfg(feature = "aws")]
            AnthropicDialect::Bedrock => crate::protocols::bedrock::decode_bedrock_error(
                ApiProfile::BedrockAnthropic,
                status,
                headers,
                body,
            ),
        }
    }

    fn new_stream_decoder(&self, _ctx: &ProtocolContext<'_>) -> Box<dyn StreamDecoder> {
        match self.dialect {
            AnthropicDialect::Direct => {
                Box::new(AnthropicStreamDecoder::new(ApiProfile::AnthropicMessages))
            }
            #[cfg(feature = "aws")]
            AnthropicDialect::Bedrock => Box::new(BedrockStreamDecoder::new()),
        }
    }

    fn new_frame_source(&self) -> Box<dyn FrameSource> {
        match self.dialect {
            AnthropicDialect::Direct => Box::new(crate::transport::sse::SseParser::new()),
            #[cfg(feature = "aws")]
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
            #[cfg(feature = "aws")]
            AnthropicDialect::Bedrock => credentials.apply_bearer(request),
        }
    }
}
