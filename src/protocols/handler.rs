use url::Url;

use super::ApiProfile;
use crate::auth::Credentials;
use crate::capabilities::ModelCapabilities;
use crate::error::{Error, Result};
use crate::request::Request;
use crate::response::{GenerateResult, Warning};
use crate::stream::{StreamEvent, StreamNormalizer};
use crate::transport::framing::{FrameSource, StreamFrame};
use crate::transport::sse::SseParser;
use crate::transport::{HeaderMap, HttpRequest, HttpResponse};

pub(crate) struct ProtocolContext<'a> {
    /// The provider's active API profile, used to attribute errors and warnings.
    pub profile: ApiProfile,
    pub model: &'a str,
    pub request: &'a Request,
    pub base_url: &'a Url,
    /// What the model accepts. Lowering fits the request to it.
    pub capabilities: &'a ModelCapabilities,
}

pub(crate) struct LoweredRequest {
    pub http: HttpRequest,
    pub warnings: Vec<Warning>,
}

pub(crate) trait ProtocolHandler: Send + Sync {
    fn lower(&self, ctx: &ProtocolContext<'_>, streaming: bool) -> Result<LoweredRequest>;

    fn decode_response(
        &self,
        ctx: &ProtocolContext<'_>,
        response: &HttpResponse,
    ) -> Result<GenerateResult>;

    fn decode_error(&self, status: u16, headers: &HeaderMap, body: &[u8]) -> Error;

    fn new_stream_decoder(&self, ctx: &ProtocolContext<'_>) -> Box<dyn StreamDecoder>;

    /// How the streamed body is split into frames. Server-Sent Events unless
    /// the profile overrides it.
    fn new_frame_source(&self) -> Box<dyn FrameSource> {
        Box::new(SseParser::new())
    }

    fn apply_auth(&self, request: &mut HttpRequest, credentials: &Credentials) -> Result<()> {
        credentials.apply_bearer(request)
    }
}

pub(crate) trait StreamDecoder: Send {
    fn on_frame(
        &mut self,
        frame: StreamFrame,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) -> Result<()>;

    fn on_eof(&mut self, normalizer: &mut StreamNormalizer, out: &mut Vec<StreamEvent>);
}

pub(crate) fn handler(profile: ApiProfile) -> &'static dyn ProtocolHandler {
    match profile {
        ApiProfile::OpenAiResponses => &super::openai::responses::Handler::OPENAI,
        ApiProfile::XaiResponses => &super::openai::responses::Handler::XAI,
        ApiProfile::OpenAiChatCompletions => &super::openai::chat::Handler::OPENAI,
        ApiProfile::XaiChatCompletions => &super::openai::chat::Handler::XAI,
        ApiProfile::OpenRouter => &super::openai::chat::Handler::OPENROUTER,
        ApiProfile::ChatGptResponses => &super::openai::responses::chatgpt::Handler,
        ApiProfile::AnthropicMessages => &super::anthropic::Handler::DIRECT,
        ApiProfile::BedrockAnthropic => &super::anthropic::Handler::BEDROCK,
        ApiProfile::GeminiGenerateContent => &super::gemini::Handler,
    }
}
