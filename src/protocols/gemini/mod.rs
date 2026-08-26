//! Gemini GenerateContent (`generativelanguage.googleapis.com`, v1beta).

use super::{LoweredRequest, ProtocolContext, ProtocolHandler, StreamDecoder};
use crate::error::{Error, Result};
use crate::response::GenerateResult;
use crate::transport::{HeaderMap, HeaderName};
use crate::transport::{HttpRequest, HttpResponse};

mod request;
mod stream;
mod types;

use self::request::lower_gemini_request;
use self::stream::GeminiStreamDecoder;
use self::types::{decode_gemini_error, decode_gemini_response};

pub(crate) struct Handler;

const SIGNATURE_KEY: &str = "thought_signature";

impl ProtocolHandler for Handler {
    fn lower(&self, ctx: &ProtocolContext<'_>, streaming: bool) -> Result<LoweredRequest> {
        lower_gemini_request(ctx, streaming)
    }

    fn decode_response(
        &self,
        _ctx: &ProtocolContext<'_>,
        response: &HttpResponse,
    ) -> Result<GenerateResult> {
        decode_gemini_response(response)
    }

    fn decode_error(&self, status: u16, headers: &HeaderMap, body: &[u8]) -> Error {
        decode_gemini_error(status, headers, body)
    }

    fn new_stream_decoder(&self, _ctx: &ProtocolContext<'_>) -> Box<dyn StreamDecoder> {
        Box::new(GeminiStreamDecoder::default())
    }

    fn apply_auth(
        &self,
        request: &mut HttpRequest,
        credentials: &crate::auth::Credentials,
    ) -> Result<()> {
        credentials.apply_native_key(request, HeaderName::from_static("x-goog-api-key"))
    }
}
