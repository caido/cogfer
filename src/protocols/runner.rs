//! Shared blocking and streaming execution.

use std::collections::VecDeque;

use bytes::Bytes;
use futures_util::StreamExt;

use super::handler::LoweredRequest;
use super::{ProtocolContext, ProtocolHandler, StreamDecoder};
use crate::auth::Rejection;
use crate::capabilities::ModelCapabilities;
use crate::error::{Error, ErrorKind, Result};
use crate::http::{enrich_error_from_headers, find_request_id, redact_headers, sanitized_url};
use crate::provider::{Authentication, Provider};
use crate::request::Request;
use crate::response::{GenerateResult, ResponseMetadata, Warning};
use crate::stream::{EventStream, StreamEvent, StreamNormalizer};
use crate::transport::framing::{FrameSource, StreamFrame};
use crate::transport::{
    HeaderMap, HeaderName, HeaderValue, HttpByteStream, HttpRequest, HttpResponse, header,
};

const TARGET: &str = "ai|runner";

struct Prepared {
    handler: &'static dyn ProtocolHandler,
    http: HttpRequest,
    warnings: Vec<Warning>,
    base_url: url::Url,
    /// The request after restriction, without the settings the model cannot
    /// accept.
    request: Request,
}

async fn prepare(
    provider: &Provider,
    model: &str,
    capabilities: &ModelCapabilities,
    request: &Request,
    streaming: bool,
) -> Result<Prepared> {
    let handler = super::handler(provider.profile());
    super::validate::validate(request, model, provider.profile(), capabilities)?;
    let (request, mut warnings) =
        super::restrict_request(request, capabilities, provider.profile());

    let base_url = provider.base_url();
    let ctx = ProtocolContext {
        profile: provider.profile(),
        model,
        request: &request,
        base_url: &base_url,
        capabilities,
    };
    let LoweredRequest {
        mut http,
        warnings: lowering_warnings,
    } = handler
        .lower(&ctx, streaming)
        .map_err(|e| annotate(e, provider, model))?;
    warnings.extend(lowering_warnings);

    apply_provider_headers(provider, &mut http);
    for (name, value) in &request.extra_headers {
        apply_header(&mut http, name, value);
    }
    authenticate(provider, handler, &mut http)
        .await
        .map_err(|error| annotate(error, provider, model))?;

    Ok(Prepared {
        handler,
        http,
        warnings,
        base_url,
        request,
    })
}

fn apply_provider_headers(provider: &Provider, http: &mut HttpRequest) {
    for (name, value) in provider.inner.config.default_headers() {
        apply_header(http, name, value);
    }
}

/// Authentication goes last so it has the final say over credential headers
/// and can sign the finished request.
async fn authenticate(
    provider: &Provider,
    handler: &dyn ProtocolHandler,
    http: &mut HttpRequest,
) -> Result<()> {
    match provider.inner.config.authentication() {
        Authentication::Credentials(credentials) => handler.apply_auth(http, credentials),
        Authentication::Authenticator(authenticator) => authenticator.authenticate(http).await,
    }
}

/// Send a buffered request, letting a refreshable authenticator recover once
/// from rejected credentials.
async fn execute(provider: &Provider, http: HttpRequest) -> Result<HttpResponse> {
    trace_wire_request(&http);
    let retry_request = retry_copy(provider, &http);
    let mut response = provider.inner.transport.execute(http).await?;
    let rejection = Rejection {
        status: response.status,
        headers: &response.headers,
    };
    if let Some(request) = authentication_retry(provider, rejection, retry_request).await? {
        trace_wire_request(&request);
        response = provider.inner.transport.execute(request).await?;
    }
    trace_wire_response(response.status, &response.headers, &response.body);
    Ok(response)
}

fn trace_wire_request(http: &HttpRequest) {
    if !log::log_enabled!(target: TARGET, log::Level::Trace) {
        return;
    }
    log::trace!(
        target: TARGET,
        "wire request: method={} url={} query_present={} headers={:?} body_bytes={}",
        http.method,
        sanitized_url(&http.url),
        http.url.query().is_some(),
        redact_headers(&http.headers),
        http.body.as_ref().map_or(0, Bytes::len),
    );
}

fn trace_wire_response(status: u16, headers: &HeaderMap, body: &[u8]) {
    if !log::log_enabled!(target: TARGET, log::Level::Trace) {
        return;
    }
    log::trace!(
        target: TARGET,
        "wire response: status={status} headers={:?} body_bytes={}",
        redact_headers(headers),
        body.len(),
    );
}

/// Set a header, comma-merging `anthropic-beta` instead of replacing so
/// caller-supplied beta flags compose with protocol-required ones.
fn apply_header(http: &mut HttpRequest, name: &HeaderName, value: &HeaderValue) {
    if name == ANTHROPIC_BETA
        && let Some(existing) = http.headers.get(name)
        && let (Ok(existing), Ok(flag)) = (existing.to_str(), value.to_str())
    {
        if !existing.split(',').any(|known| known.trim() == flag) {
            let merged = HeaderValue::from_str(&format!("{existing},{flag}"))
                .expect("joining two valid header values yields a valid header value");
            http.headers.insert(name.clone(), merged);
        }
        return;
    }
    http.headers.insert(name.clone(), value.clone());
}

const ANTHROPIC_BETA: HeaderName = HeaderName::from_static("anthropic-beta");

/// Fill in origin and model context without overwriting decoder context.
fn annotate(error: Error, provider: &Provider, model: &str) -> Error {
    let mut error = annotate_origin(error, provider);
    if error.model().is_none() {
        error = error.with_model(model);
    }
    error
}

fn annotate_origin(mut error: Error, provider: &Provider) -> Error {
    if error.origin().is_none() {
        error = error.with_origin(provider.profile().as_str());
    }
    error
}

fn retry_copy(provider: &Provider, http: &HttpRequest) -> Option<HttpRequest> {
    matches!(
        provider.inner.config.authentication(),
        Authentication::Authenticator(_)
    )
    .then(|| http.clone())
}

async fn authentication_retry(
    provider: &Provider,
    rejection: Rejection<'_>,
    request: Option<HttpRequest>,
) -> Result<Option<HttpRequest>> {
    let Some(mut request) = request else {
        return Ok(None);
    };
    // Rejected credentials: 401 from OAuth-style bearers, 403 from request
    // signers such as AWS SigV4.
    if !matches!(rejection.status, 401 | 403) {
        return Ok(None);
    }
    let Authentication::Authenticator(authenticator) = provider.inner.config.authentication()
    else {
        return Ok(None);
    };
    if authenticator
        .reauthenticate(&mut request, &rejection)
        .await?
    {
        Ok(Some(request))
    } else {
        Ok(None)
    }
}

/// Decode a buffered 2xx body, attaching request diagnostics to decoder errors.
fn decode_buffered_response(
    handler: &dyn ProtocolHandler,
    ctx: &ProtocolContext<'_>,
    mut warnings: Vec<Warning>,
    provider: &Provider,
    response: &HttpResponse,
) -> Result<GenerateResult> {
    let mut result = handler.decode_response(ctx, response).map_err(|error| {
        let error = enrich_error_from_headers(error, response.status, &response.headers);
        annotate(error, provider, ctx.model)
    })?;

    warnings.append(&mut result.warnings);
    result.warnings = warnings;
    if result.response.request_id.is_none() {
        result.response.request_id = find_request_id(&response.headers);
    }
    Ok(result)
}

pub(crate) async fn generate(
    provider: &Provider,
    model: &str,
    capabilities: &ModelCapabilities,
    request: Request,
) -> Result<GenerateResult> {
    log::debug!(target: TARGET, "generate: profile={} model={model}", provider.profile());
    let Prepared {
        handler,
        http,
        warnings,
        base_url,
        request,
    } = prepare(provider, model, capabilities, &request, false).await?;
    let response = execute(provider, http)
        .await
        .map_err(|e| annotate(e, provider, model))?;

    if !(200..300).contains(&response.status) {
        let error = annotate(
            handler.decode_error(response.status, &response.headers, &response.body),
            provider,
            model,
        );
        log::debug!(target: TARGET, "generate failed: {error}");
        return Err(error);
    }

    let ctx = ProtocolContext {
        profile: provider.profile(),
        model,
        request: &request,
        base_url: &base_url,
        capabilities,
    };
    decode_buffered_response(handler, &ctx, warnings, provider, &response)
}

pub(crate) async fn verify(provider: &Provider) -> Result<()> {
    log::debug!(target: TARGET, "verify: profile={}", provider.profile());
    let result = check_credentials(provider)
        .await
        .map_err(|error| annotate_origin(error, provider));
    if let Err(error) = &result {
        log::debug!(target: TARGET, "verify failed: {error}");
    }
    result
}

async fn check_credentials(provider: &Provider) -> Result<()> {
    let handler = super::handler(provider.profile());
    let mut http = handler.new_verify_request(&provider.base_url())?;
    apply_provider_headers(provider, &mut http);
    authenticate(provider, handler, &mut http).await?;
    let response = execute(provider, http).await?;
    if (200..300).contains(&response.status) {
        Ok(())
    } else {
        Err(handler.decode_error(response.status, &response.headers, &response.body))
    }
}

pub(crate) async fn stream(
    provider: &Provider,
    model: &str,
    capabilities: &ModelCapabilities,
    request: Request,
) -> Result<EventStream> {
    log::debug!(target: TARGET, "stream: profile={} model={model}", provider.profile());
    let Prepared {
        handler,
        http,
        warnings,
        base_url,
        request,
    } = prepare(provider, model, capabilities, &request, true).await?;
    trace_wire_request(&http);
    let retry_request = retry_copy(provider, &http);
    let mut byte_stream = provider
        .inner
        .transport
        .stream(http)
        .await
        .map_err(|e| annotate(e, provider, model))?;
    let rejection = Rejection {
        status: byte_stream.status,
        headers: &byte_stream.headers,
    };
    if let Some(request) = authentication_retry(provider, rejection, retry_request)
        .await
        .map_err(|e| annotate(e, provider, model))?
    {
        trace_wire_request(&request);
        byte_stream = provider
            .inner
            .transport
            .stream(request)
            .await
            .map_err(|e| annotate(e, provider, model))?;
    }
    if log::log_enabled!(target: TARGET, log::Level::Trace) {
        log::trace!(
            target: TARGET,
            "wire response (stream): status={} headers={:?}",
            byte_stream.status,
            redact_headers(&byte_stream.headers),
        );
    }

    if !(200..300).contains(&byte_stream.status) {
        let status = byte_stream.status;
        let headers = byte_stream.headers.clone();
        // Error bodies only need enough bytes to decode the provider envelope.
        let body = collect_error_body(byte_stream, 256 * 1024).await;
        log::trace!(target: TARGET, "wire error body: body_bytes={}", body.len());
        return Err(annotate(
            handler.decode_error(status, &headers, &body),
            provider,
            model,
        ));
    }

    let request_id = find_request_id(&byte_stream.headers);
    let mut queue: VecDeque<StreamEvent> = VecDeque::new();
    queue.push_back(StreamEvent::StreamStart { warnings });
    if let Some(request_id) = &request_id {
        queue.push_back(StreamEvent::ResponseMetadata(ResponseMetadata {
            id: None,
            model: None,
            request_id: Some(request_id.clone()),
        }));
    }
    let mut normalizer = StreamNormalizer::new(request.include_raw_events).with_error_context(
        provider.profile().as_str(),
        model,
        request_id,
    );

    // Servers that ignore `stream: true` answer with one JSON body. Decode it
    // like a blocking response and replay the result as events so callers see
    // the provider's content or error instead of a truncated stream.
    if is_json_body(&byte_stream.headers) {
        let status = byte_stream.status;
        let headers = byte_stream.headers.clone();
        // Bounded like the built-in transport's buffered responses.
        let body = collect_json_body(byte_stream, 16 * 1024 * 1024)
            .await
            .map_err(|error| {
                let error = enrich_error_from_headers(error, status, &headers);
                annotate(error, provider, model)
            })?;
        trace_wire_response(status, &headers, &body);
        let response = HttpResponse {
            status,
            headers,
            body,
        };
        // The stream start already carries the lowering warnings.
        let ctx = ProtocolContext {
            profile: provider.profile(),
            model,
            request: &request,
            base_url: &base_url,
            capabilities,
        };
        let result = decode_buffered_response(handler, &ctx, Vec::new(), provider, &response)?;
        let mut out = Vec::new();
        normalizer.replay(&mut out, result);
        queue.extend(out);
        let stream = futures_util::stream::iter(queue);
        return Ok(EventStream::new(Box::pin(stream)));
    }

    let decoder = handler.new_stream_decoder(&ProtocolContext {
        profile: provider.profile(),
        model,
        request: &request,
        base_url: &base_url,
        capabilities,
    });
    let state = StreamState {
        bytes: byte_stream.bytes,
        parser: handler.new_frame_source(),
        decoder,
        normalizer,
        queue,
        exhausted: false,
        reported_corruption: false,
    };

    let stream = futures_util::stream::unfold(state, StreamState::next_event);

    Ok(EventStream::new(Box::pin(stream)))
}

/// Whether the response declares a JSON body rather than an event stream.
fn is_json_body(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.split_once(';').map_or(value, |(mime, _)| mime))
        .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
}

struct StreamState {
    bytes: futures_util::stream::BoxStream<'static, Result<Bytes>>,
    parser: Box<dyn FrameSource>,
    decoder: Box<dyn StreamDecoder>,
    normalizer: StreamNormalizer,
    queue: VecDeque<StreamEvent>,
    exhausted: bool,
    reported_corruption: bool,
}

impl StreamState {
    async fn next_event(mut self) -> Option<(StreamEvent, Self)> {
        loop {
            if let Some(event) = self.queue.pop_front() {
                return Some((event, self));
            }
            if self.exhausted {
                return None;
            }

            let mut out = Vec::new();
            match self.bytes.next().await {
                Some(Ok(chunk)) => {
                    let frames = self.parser.push(&chunk);
                    for frame in frames {
                        if let Err(error) = self.dispatch(frame, &mut out) {
                            self.normalizer.fail(&mut out, error);
                            break;
                        }
                        if self.normalizer.is_finished() {
                            break;
                        }
                    }
                    if self.normalizer.is_finished() {
                        self.exhausted = true;
                    } else {
                        self.report_corruption(&mut out);
                    }
                }
                Some(Err(error)) => {
                    // Output may already have been consumed, so a retry is
                    // the host's decision, like a truncated stream.
                    self.normalizer.fail(&mut out, error.with_retryable(false));
                    self.exhausted = true;
                }
                None => {
                    let mut flush_failed = false;
                    // Flushing can discover corruption in a pending frame.
                    let final_frame = self.parser.finish();
                    if self.report_corruption(&mut out) {
                        self.queue.extend(out);
                        continue;
                    }
                    if let Some(frame) = final_frame
                        && let Err(error) = self.dispatch(frame, &mut out)
                    {
                        // Avoid reporting truncation after a malformed trailing frame.
                        self.normalizer.fail(&mut out, error);
                        flush_failed = true;
                    }
                    if !flush_failed {
                        if !self.normalizer.is_finished() {
                            self.decoder.on_eof(&mut self.normalizer, &mut out);
                        }
                        let error = Error::new(
                            ErrorKind::TruncatedStream,
                            "stream ended before a terminal event was received",
                        );
                        self.normalizer.on_eof(&mut out, error);
                    }
                    self.exhausted = true;
                }
            }
            self.queue.extend(out);
        }
    }

    fn dispatch(&mut self, frame: StreamFrame, out: &mut Vec<StreamEvent>) -> Result<()> {
        match frame {
            StreamFrame::Data(data) => {
                log::trace!(target: TARGET, "stream frame: data_bytes={}", data.len());
                self.decoder.on_frame(data, &mut self.normalizer, out)
            }
            #[cfg(feature = "aws")]
            StreamFrame::Exception { kind, payload } => {
                log::trace!(target: TARGET, "stream exception: kind={kind}");
                self.decoder
                    .on_exception(kind, payload, &mut self.normalizer, out);
                Ok(())
            }
        }
    }

    /// When the frame source has flagged corruption, emit the terminal error
    /// sequence once and exhaust the stream. Returns whether it fired.
    fn report_corruption(&mut self, out: &mut Vec<StreamEvent>) -> bool {
        let Some(message) = self.parser.corruption() else {
            return false;
        };
        if self.reported_corruption {
            return false;
        }
        self.reported_corruption = true;
        self.normalizer.fail(out, Error::malformed(message));
        self.exhausted = true;
        true
    }
}

async fn collect_json_body(byte_stream: HttpByteStream, limit: usize) -> Result<Bytes> {
    let mut body = Vec::new();
    let mut bytes = byte_stream.bytes;
    while let Some(chunk) = bytes.next().await {
        let chunk = chunk?;
        if chunk.len() > limit.saturating_sub(body.len()) {
            return Err(Error::malformed(format!(
                "buffered response body exceeded the configured {limit} byte limit"
            )));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(body))
}

/// Preserve the HTTP failure even when its diagnostic body cannot be read fully.
async fn collect_error_body(byte_stream: HttpByteStream, limit: usize) -> Vec<u8> {
    let mut body = Vec::new();
    let mut bytes = byte_stream.bytes;
    while let Some(chunk) = bytes.next().await {
        match chunk {
            Ok(chunk) => {
                let remaining = limit.saturating_sub(body.len());
                if remaining == 0 {
                    break;
                }
                body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
            }
            Err(error) => {
                // Preserve HTTP classification when the error body is truncated.
                log::debug!(target: TARGET, "body read failed mid-stream: {error}");
                break;
            }
        }
    }
    body
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::transport::HttpTransport;
    use crate::transport::mock::MockTransport;
    use crate::{Client, Credentials, Message, ProviderConfig};

    #[derive(Debug)]
    struct JsonTransport(Arc<MockTransport>);

    #[async_trait::async_trait]
    impl HttpTransport for JsonTransport {
        async fn execute(&self, request: HttpRequest) -> Result<HttpResponse> {
            self.0.execute(request).await
        }

        async fn stream(&self, request: HttpRequest) -> Result<HttpByteStream> {
            let mut response = self.0.stream(request).await?;
            response.headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            );
            response.headers.insert(
                HeaderName::from_static("x-request-id"),
                HeaderValue::from_static("json-request"),
            );
            response.bytes = Box::pin(response.bytes.map(|chunk| {
                chunk.map_err(|error| {
                    error.with_source(std::io::Error::other("body read interrupted"))
                })
            }));
            Ok(response)
        }
    }

    #[tokio::test]
    async fn json_stream_read_failure_preserves_transport_error() {
        for kind in [ErrorKind::Timeout, ErrorKind::Transport] {
            let mock = MockTransport::shared();
            mock.push_stream_then_error(
                vec![Bytes::from_static(
                    br#"{"choices":[{"message":{"content":"partial"#,
                )],
                kind,
            );
            let provider = Client::builder()
                .http_transport(Arc::new(JsonTransport(mock)))
                .build()
                .unwrap()
                .provider(ProviderConfig::openai_chat(Credentials::none()))
                .unwrap();

            let error = provider
                .language_model("custom-model")
                .stream(Request::builder().message(Message::user("hi")).build())
                .await
                .expect_err("a failed JSON response must fail before emitting events");

            assert_eq!(error.kind(), kind);
            assert!(error.retryable());
            assert_eq!(error.origin(), Some("openai-chat"));
            assert_eq!(error.model(), Some("custom-model"));
            assert_eq!(error.request_id(), Some("json-request"));
            assert_eq!(
                std::error::Error::source(&error).unwrap().to_string(),
                "body read interrupted"
            );
        }
    }

    #[tokio::test]
    async fn json_body_collection_rejects_overflow_without_reading_the_remainder() {
        let response = |chunks: Vec<&'static [u8]>| HttpByteStream {
            status: 200,
            headers: HeaderMap::new(),
            bytes: Box::pin(futures_util::stream::iter(
                chunks
                    .into_iter()
                    .map(|chunk| Ok(Bytes::from_static(chunk))),
            )),
        };
        assert_eq!(
            collect_json_body(response(vec![b"{", b"}"]), 2)
                .await
                .unwrap(),
            Bytes::from_static(b"{}")
        );

        for chunks in [
            vec![b"{} ".as_slice()],
            vec![b"{".as_slice(), b"} ".as_slice()],
        ] {
            let mut oversized = response(chunks);
            oversized.bytes =
                Box::pin(oversized.bytes.chain(futures_util::stream::poll_fn(|_| {
                    panic!("an oversized response must be dropped immediately")
                })));

            let error = collect_json_body(oversized, 2).await.unwrap_err();

            assert_eq!(error.kind(), ErrorKind::MalformedResponse);
            assert!(error.message().contains("byte limit"));
        }
    }

    #[test]
    fn json_bodies_are_detected_case_insensitively_with_parameters() {
        let json = HeaderMap::from_iter([(
            header::CONTENT_TYPE,
            HeaderValue::from_static("Application/JSON; charset=utf-8"),
        )]);
        let sse = HeaderMap::from_iter([(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        )]);

        assert!(is_json_body(&json));
        assert!(!is_json_body(&sse));
        assert!(!is_json_body(&HeaderMap::new()));
    }

    /// Only `application/json` exactly: other media types sharing the prefix
    /// must not be replayed as JSON bodies.
    #[test]
    fn json_lookalike_media_types_are_rejected() {
        for value in [
            "application/jsonp",
            "application/json-seq",
            "application/json5",
        ] {
            let headers = HeaderMap::from_iter([(
                header::CONTENT_TYPE,
                HeaderValue::from_str(value).unwrap(),
            )]);
            assert!(!is_json_body(&headers), "{value}");
        }
    }
}
