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
    capabilities: ModelCapabilities,
}

async fn prepare(
    provider: &Provider,
    model: &str,
    request: &Request,
    streaming: bool,
) -> Result<Prepared> {
    let handler = super::handler(provider.profile());
    super::validate::validate(request, model, provider.profile())?;

    let base_url = provider.base_url();
    let capabilities = provider.capabilities();
    let ctx = ProtocolContext {
        profile: provider.profile(),
        model,
        request,
        base_url: &base_url,
        capabilities: &capabilities,
    };
    let LoweredRequest { mut http, warnings } = handler
        .lower(&ctx, streaming)
        .map_err(|e| annotate(e, provider, model))?;

    // Merge anthropic-beta so user flags preserve protocol requirements.
    for (name, value) in provider.inner.config.default_headers() {
        apply_header(&mut http, name, value);
    }
    for (name, value) in &request.extra_headers {
        apply_header(&mut http, name, value);
    }

    match provider.inner.config.authentication() {
        Authentication::Credentials(credentials) => handler
            .apply_auth(&mut http, credentials)
            .map_err(|error| annotate(error, provider, model))?,
        Authentication::Authenticator(authenticator) => {
            authenticator
                .authenticate(&mut http)
                .await
                .map_err(|error| annotate(error, provider, model))?;
        }
    }

    Ok(Prepared {
        handler,
        http,
        warnings,
        base_url,
        capabilities,
    })
}

fn trace_wire_request(http: &HttpRequest) {
    if !log::log_enabled!(target: TARGET, log::Level::Trace) {
        return;
    }
    let header_names: Vec<_> = http.headers.keys().map(HeaderName::as_str).collect();
    log::trace!(
        target: TARGET,
        "wire request: method={} url={} query_present={} headers={header_names:?} body_bytes={}",
        http.method,
        sanitized_url(&http.url),
        http.url.query().is_some(),
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
fn annotate(mut error: Error, provider: &Provider, model: &str) -> Error {
    if error.origin().is_none() {
        error = error.with_origin(provider.profile().as_str());
    }
    if error.model().is_none() {
        error = error.with_model(model);
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
    request: Request,
) -> Result<GenerateResult> {
    log::debug!(target: TARGET, "generate: profile={} model={model}", provider.profile());
    let Prepared {
        handler,
        http,
        warnings,
        base_url,
        capabilities,
    } = prepare(provider, model, &request, false).await?;
    trace_wire_request(&http);
    let retry_request = retry_copy(provider, &http);
    let mut response = provider
        .inner
        .transport
        .execute(http)
        .await
        .map_err(|e| annotate(e, provider, model))?;
    let rejection = Rejection {
        status: response.status,
        headers: &response.headers,
    };
    if let Some(request) = authentication_retry(provider, rejection, retry_request)
        .await
        .map_err(|e| annotate(e, provider, model))?
    {
        trace_wire_request(&request);
        response = provider
            .inner
            .transport
            .execute(request)
            .await
            .map_err(|e| annotate(e, provider, model))?;
    }
    trace_wire_response(response.status, &response.headers, &response.body);

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
        capabilities: &capabilities,
    };
    decode_buffered_response(handler, &ctx, warnings, provider, &response)
}

pub(crate) async fn stream(
    provider: &Provider,
    model: &str,
    request: Request,
) -> Result<EventStream> {
    log::debug!(target: TARGET, "stream: profile={} model={model}", provider.profile());
    let Prepared {
        handler,
        http,
        warnings,
        base_url,
        capabilities,
    } = prepare(provider, model, &request, true).await?;
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
        let body = collect_body(byte_stream, 256 * 1024).await;
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
        let body = collect_body(byte_stream, 16 * 1024 * 1024).await;
        trace_wire_response(status, &headers, &body);
        let response = HttpResponse {
            status,
            headers,
            body: Bytes::from(body),
        };
        // The stream start already carries the lowering warnings.
        let ctx = ProtocolContext {
            profile: provider.profile(),
            model,
            request: &request,
            base_url: &base_url,
            capabilities: &capabilities,
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
        capabilities: &capabilities,
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
        .is_some_and(|value| {
            value
                .trim_start()
                .to_ascii_lowercase()
                .starts_with("application/json")
        })
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
                    // Flushing can discover invalid UTF-8 in an unterminated line.
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

    /// Hand one frame to the decoder.
    fn dispatch(&mut self, frame: StreamFrame, out: &mut Vec<StreamEvent>) -> Result<()> {
        match frame {
            StreamFrame::Data(data) => {
                log::trace!(target: TARGET, "stream frame: data_bytes={}", data.len());
                self.decoder.on_frame(data, &mut self.normalizer, out)
            }
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

async fn collect_body(byte_stream: HttpByteStream, limit: usize) -> Vec<u8> {
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
    use super::*;

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
}
