#[test]
fn credential_headers_are_redacted_in_debug() {
    use llmwire::transport::{HeaderName, HeaderValue, HttpRequest};

    let mut request = HttpRequest::post_json(
        "https://example.com".parse().unwrap(),
        &serde_json::json!({"prompt": "secret"}),
    )
    .unwrap();
    request.headers.insert(
        HeaderName::from_static("x-custom-auth"),
        HeaderValue::from_static("super-secret"),
    );

    let debug = format!("{request:?}");
    assert!(!debug.contains("super-secret"), "{debug}");
    assert!(debug.contains("<redacted>"), "{debug}");
    assert!(debug.contains("application/json"), "{debug}");
}

#[test]
fn request_debug_redacts_url_credentials_and_query() {
    use llmwire::transport::HttpRequest;

    let request = HttpRequest::post_json(
        "https://user:password@example.com/v1?api_key=secret#fragment"
            .parse()
            .unwrap(),
        &serde_json::json!({"prompt": "hello"}),
    )
    .unwrap();

    let debug = format!("{request:?}");
    assert!(!debug.contains("user"), "{debug}");
    assert!(!debug.contains("password"), "{debug}");
    assert!(!debug.contains("api_key"), "{debug}");
    assert!(!debug.contains("secret"), "{debug}");
    assert!(!debug.contains("fragment"), "{debug}");
    assert!(debug.contains("https://example.com/v1"), "{debug}");
}

#[test]
fn provider_debug_redacts_url_credentials_and_query() {
    let config = llmwire::ProviderConfig::new(
        llmwire::ApiProfile::OpenAiResponses,
        llmwire::Credentials::api_key("k"),
    )
    .with_base_url(
        "https://user:password@example.com/v1?api_key=secret#fragment"
            .parse()
            .unwrap(),
    );

    let debug = format!("{config:?}");
    assert!(!debug.contains("user"), "{debug}");
    assert!(!debug.contains("password"), "{debug}");
    assert!(!debug.contains("api_key"), "{debug}");
    assert!(!debug.contains("secret"), "{debug}");
    assert!(!debug.contains("fragment"), "{debug}");
    assert!(debug.contains("https://example.com/v1"), "{debug}");
}

#[test]
fn provider_tool_assistant_part_has_an_unambiguous_wire_discriminator() {
    let part = llmwire::AssistantPart::ProviderTool {
        provider_tool: llmwire::ProviderToolPart {
            id: Some("provider-tool-1".into()),
            kind: "web_search_call".into(),
            namespace: "openai".into(),
            payload: serde_json::json!({"status": "completed"}),
        },
    };

    let value = serde_json::to_value(&part).unwrap();
    assert_eq!(value["kind"], "provider-tool");
    assert_eq!(value["provider_tool"]["kind"], "web_search_call");
    assert_eq!(
        serde_json::from_value::<llmwire::AssistantPart>(value).unwrap(),
        part
    );
}

#[test]
fn secret_string_debug_is_redacted() {
    let secret = llmwire::SecretString::new("sk-live-123");
    assert_eq!(format!("{secret:?}"), "SecretString(<redacted>)");
}

#[test]
fn secret_string_display_is_redacted() {
    let secret = llmwire::SecretString::new("sk-live-123");
    assert_eq!(format!("{secret}"), "<redacted>");
}

#[test]
fn credentials_debug_redacts_nested_secrets() {
    let credentials = llmwire::Credentials::api_key("sk-live-123");
    assert_eq!(
        format!("{credentials:?}"),
        "ApiKey(SecretString(<redacted>))"
    );
}

#[test]
fn null_provider_option_does_not_erase_the_body() {
    let mut metadata = llmwire::ProviderMetadata::new();
    metadata.merge(llmwire::ProviderMetadata::with(
        "openai",
        serde_json::Value::Null,
    ));
    assert!(
        metadata.get("openai").is_none(),
        "null namespace must not be stored"
    );
}

#[tokio::test]
async fn provider_options_must_be_objects() {
    use llmwire::transport::mock::MockTransport;
    use llmwire::{Credentials, ErrorKind, Message, ProviderConfig, ProviderMetadata, Request};

    let mock = MockTransport::shared();
    let provider = llmwire::Client::builder()
        .http_transport(mock.clone())
        .build()
        .unwrap()
        .provider(ProviderConfig::openai_chat(Credentials::api_key("k")))
        .unwrap();
    let mut options = std::collections::BTreeMap::new();
    options.insert("openai".to_string(), serde_json::Value::Null);
    let request = Request {
        provider_options: ProviderMetadata::from(options),
        ..Request::builder().message(Message::user("hi")).build()
    };

    let error = provider
        .language_model("gpt-5.6")
        .generate(request)
        .await
        .expect_err("a null namespace would replace the wire body");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert!(
        error.message().contains("provider_options.openai"),
        "{error}"
    );
    assert!(mock.requests().is_empty());
}

/// A request signer that stamps a generation counter, standing in for a
/// signature that must be recomputed when the provider rejects it.
#[derive(Debug, Default)]
struct CountingSigner {
    signatures: std::sync::atomic::AtomicU32,
}

#[async_trait::async_trait]
impl llmwire::RequestAuthenticator for CountingSigner {
    async fn authenticate(
        &self,
        request: &mut llmwire::transport::HttpRequest,
    ) -> llmwire::Result<()> {
        let generation = self
            .signatures
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        request.headers.insert(
            llmwire::transport::HeaderName::from_static("x-signature"),
            llmwire::transport::HeaderValue::from_str(&format!("v{generation}")).unwrap(),
        );
        Ok(())
    }

    async fn reauthenticate(
        &self,
        request: &mut llmwire::transport::HttpRequest,
        rejection: &llmwire::Rejection<'_>,
    ) -> llmwire::Result<bool> {
        if rejection.status != 403 {
            return Ok(false);
        }
        self.authenticate(request).await?;
        Ok(true)
    }
}

#[tokio::test]
async fn signers_recover_from_a_forbidden_response() {
    use llmwire::transport::mock::MockTransport;
    use llmwire::{Credentials, ProviderConfig};

    let mock = MockTransport::shared();
    mock.push_json(403, &serde_json::json!({"message": "signature expired"}));
    mock.push_json(
        200,
        &serde_json::json!({
            "id": "resp_1", "object": "response", "status": "completed", "model": "m",
            "output": [{"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        }),
    );
    let provider = crate::common::provider_with(
        &mock,
        ProviderConfig::openai_responses(Credentials::none())
            .with_authenticator(std::sync::Arc::new(CountingSigner::default())),
    );

    let result = provider
        .language_model("m")
        .generate(crate::common::text_request("hi"))
        .await
        .expect("the re-signed request succeeds");

    assert_eq!(result.text(), "ok");
    let requests = mock.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].method, llmwire::transport::Method::POST);
    assert_eq!(
        crate::common::header(&requests[0], "x-signature"),
        Some("v1")
    );
    assert_eq!(
        crate::common::header(&requests[1], "x-signature"),
        Some("v2")
    );
}
