#[test]
fn credential_headers_are_redacted_in_debug() {
    use caido_ai::transport::HttpRequest;

    let mut request = HttpRequest::post_json(
        "https://example.com".parse().unwrap(),
        &serde_json::json!({"prompt": "secret"}),
    )
    .unwrap();
    request.set_header("x-custom-auth", "super-secret");
    request.set_header("content-type", "application/json");

    let debug = format!("{request:?}");
    assert!(!debug.contains("super-secret"), "{debug}");
    assert!(debug.contains("<redacted>"), "{debug}");
    assert!(debug.contains("application/json"), "{debug}");
}

#[test]
fn request_debug_redacts_url_credentials_and_query() {
    use caido_ai::transport::HttpRequest;

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
    let config = caido_ai::ProviderConfig::new(
        caido_ai::ApiProfile::OpenAiResponses,
        caido_ai::Credentials::api_key("k"),
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
    let part = caido_ai::AssistantPart::ProviderTool {
        provider_tool: caido_ai::ProviderToolPart {
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
        serde_json::from_value::<caido_ai::AssistantPart>(value).unwrap(),
        part
    );
}

#[test]
fn secret_string_debug_is_redacted() {
    let secret = caido_ai::SecretString::new("sk-live-123");
    assert_eq!(format!("{secret:?}"), "SecretString(<redacted>)");
}

#[test]
fn secret_string_display_is_redacted() {
    let secret = caido_ai::SecretString::new("sk-live-123");
    assert_eq!(format!("{secret}"), "<redacted>");
}

#[test]
fn credentials_debug_redacts_nested_secrets() {
    let credentials = caido_ai::Credentials::api_key("sk-live-123");
    assert_eq!(
        format!("{credentials:?}"),
        "ApiKey(SecretString(<redacted>))"
    );
}

#[test]
fn null_provider_option_does_not_erase_the_body() {
    let mut metadata = caido_ai::ProviderMetadata::new();
    metadata.merge(caido_ai::ProviderMetadata::with(
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
    use caido_ai::transport::mock::MockTransport;
    use caido_ai::{Credentials, ErrorKind, Message, ProviderConfig, ProviderMetadata, Request};

    let mock = MockTransport::shared();
    let provider = caido_ai::Client::builder()
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
