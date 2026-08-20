use super::*;

#[tokio::test]
async fn authenticator_sets_bearer_and_account_headers() {
    let mock = MockTransport::shared();
    mock.push_sse(&completed_transcript());
    let auth_transport = MockTransport::shared();

    let tokens = ChatGptTokens::new(jwt(now() + 3600, "acct_42"));
    let authenticator =
        ChatGptAuthenticator::new(tokens, ChatGptOAuth::new(auth_transport.clone()));
    let provider = provider_with(
        &mock,
        ProviderConfig::chatgpt(Credentials::none()).with_authenticator(Arc::new(authenticator)),
    );

    provider
        .language_model("gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");

    let http = &mock.requests()[0];
    let authorization = http
        .headers
        .iter()
        .find(|(name, _)| name == "authorization")
        .map(|(_, value)| value.clone())
        .expect("authorization header");
    assert!(authorization.starts_with("Bearer eyJ") || authorization.starts_with("Bearer "));
    assert!(
        http.headers
            .iter()
            .any(|(name, value)| name == "chatgpt-account-id" && value == "acct_42"),
        "account id derived from the JWT should be sent"
    );
    assert!(
        auth_transport.requests().is_empty(),
        "a fresh token must not trigger a refresh"
    );
}

#[tokio::test]
async fn authenticator_refreshes_expired_tokens_and_reports_them() {
    let mock = MockTransport::shared();
    mock.push_sse(&completed_transcript());

    let fresh_access = jwt(now() + 3600, "acct_new");
    let auth_transport = MockTransport::shared();
    auth_transport.push_json(
        200,
        &json!({
            "access_token": fresh_access,
            "refresh_token": "refresh-2",
        }),
    );

    let store = Arc::new(RecordingTokenStore::default());
    let authenticator = ChatGptAuthenticator::new(
        ChatGptTokens::new(jwt(now() - 10, "acct_old")).with_refresh_token("refresh-1"),
        ChatGptOAuth::new(auth_transport.clone()),
    )
    .with_token_store(store.clone());

    let provider = provider_with(
        &mock,
        ProviderConfig::chatgpt(Credentials::none()).with_authenticator(Arc::new(authenticator)),
    );
    provider
        .language_model("gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds after refresh");

    let refresh_request = &auth_transport.requests()[0];
    assert_eq!(
        refresh_request.url.as_str(),
        "https://auth.openai.com/oauth/token"
    );
    let form = String::from_utf8(refresh_request.body.clone().unwrap().to_vec()).unwrap();
    assert!(form.contains("grant_type=refresh_token"));
    assert!(form.contains("refresh_token=refresh-1"));

    let http = &mock.requests()[0];
    assert!(
        http.headers
            .iter()
            .any(|(name, value)| name == "authorization"
                && *value == format!("Bearer {fresh_access}"))
    );
    assert!(
        http.headers
            .iter()
            .any(|(name, value)| name == "chatgpt-account-id" && value == "acct_new")
    );

    let saved = store.saved().expect("refreshed tokens persisted");
    assert_eq!(saved.access_token, fresh_access);
    assert_eq!(saved.refresh_token.as_deref(), Some("refresh-2"));
    assert_eq!(saved.account_id.as_deref(), Some("acct_new"));
}

#[tokio::test]
async fn invalid_grant_refresh_is_an_authentication_error() {
    let mock = MockTransport::shared();
    let auth_transport = MockTransport::shared();
    auth_transport.push_json(400, &json!({"error": "invalid_grant"}));

    let authenticator = Arc::new(ChatGptAuthenticator::new(
        ChatGptTokens::new(jwt(now() - 10, "acct")).with_refresh_token("refresh-dead"),
        ChatGptOAuth::new(auth_transport.clone()),
    ));
    let provider = provider_with(
        &mock,
        ProviderConfig::chatgpt(Credentials::none()).with_authenticator(authenticator.clone()),
    );

    let error = provider
        .language_model("gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect_err("dead refresh token should fail the request");
    assert_eq!(error.kind(), ErrorKind::Authentication);
    assert_eq!(authenticator.status().await, OAuthStatus::ReauthRequired);
    provider
        .language_model("gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect_err("reauthentication state remains terminal");
    assert_eq!(auth_transport.requests().len(), 1);
    assert!(mock.requests().is_empty(), "no API request should be sent");
}

#[tokio::test]
async fn refreshed_tokens_are_not_published_when_persistence_fails() {
    let mock = MockTransport::shared();
    let auth_transport = MockTransport::shared();
    let fresh_access = jwt(now() + 3600, "acct_new");
    auth_transport.push_json(
        200,
        &json!({
            "access_token": fresh_access,
            "refresh_token": "refresh-2",
        }),
    );

    let old_access = jwt(now() - 10, "acct_old");
    let store = Arc::new(RecordingTokenStore::<ChatGptTokens>::failing());
    let authenticator = Arc::new(
        ChatGptAuthenticator::new(
            ChatGptTokens::new(old_access.clone()).with_refresh_token("refresh-1"),
            ChatGptOAuth::new(auth_transport),
        )
        .with_token_store(store),
    );
    let provider = provider_with(
        &mock,
        ProviderConfig::chatgpt(Credentials::none()).with_authenticator(authenticator.clone()),
    );

    let error = provider
        .language_model("gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect_err("a failed durable save must fail authentication");

    assert_eq!(error.message(), "token store failed");
    assert_eq!(authenticator.tokens().await.access_token, old_access);
    assert_eq!(authenticator.status().await, OAuthStatus::PersistenceFailed);
    assert!(mock.requests().is_empty());
}

#[tokio::test]
async fn unauthorized_response_refreshes_and_retries_before_output() {
    let mock = MockTransport::shared();
    mock.push_json(401, &json!({"error": {"message": "expired"}}));
    mock.push_sse(&completed_transcript());

    let fresh_access = jwt(now() + 3600, "acct_new");
    let auth_transport = MockTransport::shared();
    auth_transport.push_json(
        200,
        &json!({
            "access_token": fresh_access,
            "refresh_token": "refresh-2",
        }),
    );
    let tokens = ChatGptTokens::new(jwt(now() + 3600, "acct_old")).with_refresh_token("refresh-1");
    let authenticator =
        ChatGptAuthenticator::new(tokens, ChatGptOAuth::new(auth_transport.clone()));
    let provider = provider_with(
        &mock,
        ProviderConfig::chatgpt(Credentials::none()).with_authenticator(Arc::new(authenticator)),
    );

    provider
        .language_model("gpt-5.6-sol")
        .generate(text_request("hi"))
        .await
        .expect("one pre-output authentication retry succeeds");

    let requests = mock.requests();
    assert_eq!(requests.len(), 2);
    assert_ne!(
        header(&requests[0], "authorization"),
        header(&requests[1], "authorization")
    );
    assert_eq!(auth_transport.requests().len(), 1);
}

#[tokio::test]
async fn unauthorized_stream_refreshes_before_exposing_events() {
    let mock = MockTransport::shared();
    mock.push_json(401, &json!({"error": {"message": "expired"}}));
    mock.push_sse(&completed_transcript());

    let auth_transport = MockTransport::shared();
    auth_transport.push_json(
        200,
        &json!({
            "access_token": jwt(now() + 3600, "acct_new"),
            "refresh_token": "refresh-2",
        }),
    );
    let tokens = ChatGptTokens::new(jwt(now() + 3600, "acct_old")).with_refresh_token("refresh-1");
    let authenticator =
        ChatGptAuthenticator::new(tokens, ChatGptOAuth::new(auth_transport.clone()));
    let provider = provider_with(
        &mock,
        ProviderConfig::chatgpt(Credentials::none()).with_authenticator(Arc::new(authenticator)),
    );

    let events = drain(
        provider
            .language_model("gpt-5.6-sol")
            .stream(text_request("hi"))
            .await
            .expect("stream establishes after authentication recovery"),
    )
    .await;

    assert!(matches!(
        events.first(),
        Some(StreamEvent::StreamStart { .. })
    ));
    assert_eq!(mock.requests().len(), 2);
    assert_eq!(auth_transport.requests().len(), 1);
}
