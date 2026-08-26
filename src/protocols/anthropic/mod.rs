//! Anthropic Messages API (`POST {base}/messages`).

use super::{LoweredRequest, ProtocolContext, ProtocolHandler, StreamDecoder};
use crate::error::{Error, Result};
use crate::response::GenerateResult;
use crate::transport::{HeaderMap, HeaderName};
use crate::transport::{HttpRequest, HttpResponse};

mod request;
mod stream;
mod types;

use self::request::lower_anthropic_request;
use self::stream::AnthropicStreamDecoder;
use self::types::{decode_anthropic_error, decode_anthropic_response};

pub(crate) struct Handler;

impl ProtocolHandler for Handler {
    fn lower(&self, ctx: &ProtocolContext<'_>, streaming: bool) -> Result<LoweredRequest> {
        lower_anthropic_request(ctx, streaming)
    }

    fn decode_response(
        &self,
        _ctx: &ProtocolContext<'_>,
        response: &HttpResponse,
    ) -> Result<GenerateResult> {
        decode_anthropic_response(response)
    }

    fn decode_error(&self, status: u16, headers: &HeaderMap, body: &[u8]) -> Error {
        decode_anthropic_error(status, headers, body)
    }

    fn new_stream_decoder(&self, _ctx: &ProtocolContext<'_>) -> Box<dyn StreamDecoder> {
        Box::new(AnthropicStreamDecoder::default())
    }

    fn apply_auth(
        &self,
        request: &mut HttpRequest,
        credentials: &crate::auth::Credentials,
    ) -> Result<()> {
        credentials.apply_native_key(request, HeaderName::from_static("x-api-key"))
    }
}
