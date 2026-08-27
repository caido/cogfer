use serde_json::{Value, json};

use super::types::{
    GeminiCandidate, GeminiFunctionCall, GeminiPart, GenerateContentResponse, decode_function_call,
    error_finish_error, finish_from_reason, signature_metadata,
};
use crate::error::{Error, Result};
use crate::message::{ReasoningContent, ReasoningPart};
use crate::metadata::ProviderMetadata;
use crate::protocols::StreamDecoder;
use crate::response::{Finish, FinishReason, ResponseMetadata};
use crate::stream::{Citation, StreamEvent, StreamNormalizer};

const TEXT_BLOCK: &str = "t0";
const REASONING_BLOCK: &str = "r0";

#[derive(Default)]
pub(crate) struct GeminiStreamDecoder {
    text_open: bool,
    reasoning_open: bool,
    reasoning_content: Vec<ReasoningContent>,
    reasoning_metadata: ProviderMetadata,
    finish: Option<Finish>,
    response_id_sent: bool,
    response_model_sent: bool,
    /// Number of cumulative grounding chunks already emitted as citations.
    citations_emitted: usize,
}

impl GeminiStreamDecoder {
    fn close_text(&mut self, normalizer: &mut StreamNormalizer, out: &mut Vec<StreamEvent>) {
        if self.text_open {
            self.text_open = false;
            normalizer.end_text(out, TEXT_BLOCK);
        }
    }

    fn close_reasoning(&mut self, normalizer: &mut StreamNormalizer, out: &mut Vec<StreamEvent>) {
        if self.reasoning_open {
            self.reasoning_open = false;
            let part = ReasoningPart {
                id: None,
                content: std::mem::take(&mut self.reasoning_content),
                provider_metadata: std::mem::take(&mut self.reasoning_metadata),
            };
            normalizer.end_reasoning(out, REASONING_BLOCK, Some(part));
        }
    }

    fn finish_error_frame(
        &mut self,
        error: Error,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        normalizer.error(out, error);
        self.close_reasoning(normalizer, out);
        self.close_text(normalizer, out);
        normalizer.finish(out, Finish::with_raw(FinishReason::Error, "error"));
    }

    fn emit_response_metadata(
        &mut self,
        chunk: &GenerateContentResponse,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        let id = (!self.response_id_sent)
            .then(|| chunk.response_id.clone())
            .flatten();
        let model = (!self.response_model_sent)
            .then(|| chunk.model_version.clone())
            .flatten();
        self.response_id_sent |= id.is_some();
        self.response_model_sent |= model.is_some();
        if id.is_some() || model.is_some() {
            normalizer.metadata(
                out,
                ResponseMetadata {
                    id,
                    model,
                    request_id: None,
                },
            );
        }
    }

    fn record_prompt_feedback(
        &mut self,
        chunk: &GenerateContentResponse,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        if let Some(feedback) = &chunk.prompt_feedback {
            normalizer.provider_metadata(out, feedback.provider_metadata());
            if let Some(reason) = &feedback.block_reason {
                self.finish = Some(Finish::with_raw(
                    FinishReason::ContentFilter,
                    reason.clone(),
                ));
            }
        }
    }

    fn emit_grounding(
        &mut self,
        grounding: &Value,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        normalizer.provider_metadata(
            out,
            ProviderMetadata::with("gemini", json!({"groundingMetadata": grounding})),
        );
        let Some(chunks) = grounding.get("groundingChunks").and_then(Value::as_array) else {
            return;
        };
        for source_chunk in chunks.iter().skip(self.citations_emitted) {
            let source = source_chunk
                .get("web")
                .or_else(|| source_chunk.get("retrievedContext"))
                .unwrap_or(source_chunk);
            let text_of = |key: &str| source.get(key).and_then(Value::as_str).map(str::to_string);
            normalizer.citation(
                out,
                None,
                Citation {
                    url: text_of("uri"),
                    title: text_of("title"),
                    cited_text: None,
                    raw: source_chunk.clone(),
                },
            );
        }
        self.citations_emitted = self.citations_emitted.max(chunks.len());
    }

    fn emit_reasoning_part(
        &mut self,
        part: &GeminiPart,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        self.close_text(normalizer, out);
        self.reasoning_open = true;
        if let Some(text) = &part.text {
            self.reasoning_content
                .push(ReasoningContent::Summary { text: text.clone() });
            normalizer.reasoning_delta(out, REASONING_BLOCK.into(), text.clone());
        }
        if let Some(signature) = &part.thought_signature {
            self.reasoning_metadata.merge(signature_metadata(signature));
        }
    }

    fn emit_tool_call(
        &mut self,
        part: &GeminiPart,
        function_call: &GeminiFunctionCall,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        self.close_reasoning(normalizer, out);
        self.close_text(normalizer, out);
        let call = decode_function_call(part, function_call);
        normalizer.start_tool(out, call.call_id.clone(), call.name, call.item_id);
        normalizer.tool_metadata(&call.call_id, call.provider_metadata);
        normalizer.tool_delta(out, &call.call_id, call.arguments);
        normalizer.complete_tool(&call.call_id, None);
    }

    fn emit_text_part(
        &mut self,
        part: &GeminiPart,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        if let Some(text) = &part.text {
            if text.is_empty() && part.thought_signature.is_none() {
                return;
            }
            self.close_reasoning(normalizer, out);
            self.text_open = true;
            normalizer.text_delta(out, TEXT_BLOCK.into(), text.clone());
            if let Some(signature) = &part.thought_signature {
                normalizer.text_metadata(TEXT_BLOCK, signature_metadata(signature));
            }
        } else if let Some(signature) = &part.thought_signature {
            self.close_reasoning(normalizer, out);
            self.text_open = true;
            normalizer.start_text(out, TEXT_BLOCK.into());
            normalizer.text_metadata(TEXT_BLOCK, signature_metadata(signature));
        }
    }

    fn emit_part(
        &mut self,
        part: &GeminiPart,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        if part.thought == Some(true) {
            self.emit_reasoning_part(part, normalizer, out);
        } else if let Some(function_call) = &part.function_call {
            self.emit_tool_call(part, function_call, normalizer, out);
        } else {
            self.emit_text_part(part, normalizer, out);
        }
    }

    fn emit_candidate(
        &mut self,
        candidate: &GeminiCandidate,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) {
        if let Some(grounding) = &candidate.grounding_metadata {
            self.emit_grounding(grounding, normalizer, out);
        }
        if let Some(citations) = &candidate.citation_metadata {
            normalizer.provider_metadata(
                out,
                ProviderMetadata::with("gemini", json!({"citationMetadata": citations})),
            );
        }
        if let Some(metadata) = candidate.safety_metadata() {
            normalizer.provider_metadata(out, metadata);
        }
        if let Some(parts) = candidate
            .content
            .as_ref()
            .and_then(|content| content.parts.as_ref())
        {
            for part in parts {
                self.emit_part(part, normalizer, out);
            }
        }
        if let Some(reason) = &candidate.finish_reason {
            self.finish = Some(finish_from_reason(Some(reason)));
        }
    }
}

impl StreamDecoder for GeminiStreamDecoder {
    fn on_frame(
        &mut self,
        data: String,
        normalizer: &mut StreamNormalizer,
        out: &mut Vec<StreamEvent>,
    ) -> Result<()> {
        // Type mismatches fail the stream: demoting them would silently lose
        // candidates or the finish reason.
        let chunk: GenerateContentResponse = serde_json::from_str(&data)
            .map_err(|e| Error::malformed(format!("gemini: invalid stream chunk: {e}")))?;
        if let Some(error) = chunk.error {
            self.finish_error_frame(error.into_error(None), normalizer, out);
            return Ok(());
        }

        self.emit_response_metadata(&chunk, normalizer, out);
        if let Some(usage) = &chunk.usage_metadata {
            normalizer.merge_usage(&usage.to_usage());
        }
        self.record_prompt_feedback(&chunk, normalizer, out);
        if let Some(candidate) = chunk.candidates.as_ref().and_then(|c| c.first()) {
            self.emit_candidate(candidate, normalizer, out);
        }
        Ok(())
    }

    fn on_eof(&mut self, normalizer: &mut StreamNormalizer, out: &mut Vec<StreamEvent>) {
        // Gemini uses EOF as its terminal sentinel.
        if let Some(finish) = self.finish.take() {
            self.close_reasoning(normalizer, out);
            self.close_text(normalizer, out);
            // Error finish reasons must also fail the normalized stream.
            if finish.reason == FinishReason::Error {
                normalizer.error(out, error_finish_error(&finish));
            }
            normalizer.finish(out, finish);
        }
    }
}
