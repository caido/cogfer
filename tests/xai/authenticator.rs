use super::*;

#[tokio::test]
async fn refresh_keeps_an_unrotated_refresh_token() {
    let transport = MockTransport::shared();
    transport.push_json(
        200,
        &json!({"access_token": "access-2", "expires_in": 3600}),
    );

    let tokens = XaiOAuth::new(transport.clone())
        .refresh("refresh-1")
        .await
        .expect("refresh succeeds");
    assert_eq!(tokens.access_token, "access-2");
    assert_eq!(tokens.refresh_token.as_deref(), Some("refresh-1"));

    let request = &transport.requests()[0];
    assert_eq!(request.url.as_str(), "https://auth.x.ai/oauth2/token");
    let form = form_body(request);
    assert!(form.contains("grant_type=refresh_token"));
    assert!(form.contains("refresh_token=refresh-1"));
}

#[tokio::test]
async fn authenticator_sends_bearer_without_needless_refresh() {
    let mock = MockTransport::shared();
    mock.push_json(200, &responses_completed("grok-4.5"));
    let auth_transport = MockTransport::shared();

    let mut tokens = XaiTokens::new("fresh-access");
    tokens.expires_at = Some(now() + 3600);
    let authenticator = XaiAuthenticator::new(tokens, XaiOAuth::new(auth_transport.clone()));
    let provider = provider_with(
        &mock,
        ProviderConfig::xai(Credentials::header(
            HeaderName::from_static("x-static-auth"),
            "stale",
        ))
        .with_authenticator(Arc::new(authenticator)),
    );

    provider
        .language_model("grok-4.5")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds");

    let http = &mock.requests()[0];
    assert_eq!(header(http, "authorization"), Some("Bearer fresh-access"));
    assert_eq!(header(http, "x-static-auth"), None);
    assert!(
        auth_transport.requests().is_empty(),
        "a fresh token must not trigger a refresh"
    );
}

#[tokio::test]
async fn authenticator_refreshes_expiring_tokens_and_reports_them() {
    let mock = MockTransport::shared();
    mock.push_json(200, &responses_completed("grok-4.5"));

    let auth_transport = MockTransport::shared();
    auth_transport.push_json(
        200,
        &json!({
            "access_token": "access-2",
            "refresh_token": "refresh-2",
            "expires_in": 3600,
        }),
    );

    let mut tokens = XaiTokens::new("access-1").with_refresh_token("refresh-1");
    tokens.expires_at = Some(now() + 60);

    let store = Arc::new(RecordingTokenStore::default());
    let authenticator = XaiAuthenticator::new(tokens, XaiOAuth::new(auth_transport.clone()))
        .with_token_store(store.clone());

    let provider = provider_with(
        &mock,
        ProviderConfig::xai(Credentials::none()).with_authenticator(Arc::new(authenticator)),
    );
    provider
        .language_model("grok-4.5")
        .generate(text_request("hi"))
        .await
        .expect("generate succeeds after refresh");

    let refresh_request = &auth_transport.requests()[0];
    assert_eq!(
        refresh_request.url.as_str(),
        "https://auth.x.ai/oauth2/token"
    );
    let form = form_body(refresh_request);
    assert!(form.contains("grant_type=refresh_token"));
    assert!(form.contains("refresh_token=refresh-1"));

    let http = &mock.requests()[0];
    assert_eq!(header(http, "authorization"), Some("Bearer access-2"));

    let saved = store.saved().expect("refreshed tokens persisted");
    assert_eq!(saved.access_token, "access-2");
    assert_eq!(saved.refresh_token.as_deref(), Some("refresh-2"));
}

#[tokio::test]
async fn short_lived_refreshed_token_is_used_without_refreshing_again() {
    let mock = MockTransport::shared();
    mock.push_json(200, &responses_completed("grok-4.5"));
    let auth_transport = MockTransport::shared();
    auth_transport.push_json(
        200,
        &json!({
            "access_token": "access-2",
            "refresh_token": "refresh-2",
            "expires_in": 60,
        }),
    );
    let mut tokens = XaiTokens::new("access-1").with_refresh_token("refresh-1");
    tokens.expires_at = Some(now() - 1);
    let authenticator = XaiAuthenticator::new(tokens, XaiOAuth::new(auth_transport.clone()));
    let provider = provider_with(
        &mock,
        ProviderConfig::xai(Credentials::none()).with_authenticator(Arc::new(authenticator)),
    );

    provider
        .language_model("grok-4.5")
        .generate(text_request("hi"))
        .await
        .expect("the first refreshed token is used for this request");

    assert_eq!(auth_transport.requests().len(), 1);
}

#[tokio::test]
async fn invalid_grant_refresh_is_an_authentication_error() {
    let mock = MockTransport::shared();
    let auth_transport = MockTransport::shared();
    auth_transport.push_json(400, &json!({"error": "invalid_grant"}));

    let mut tokens = XaiTokens::new("access-old").with_refresh_token("refresh-dead");
    tokens.expires_at = Some(now() - 10);
    let authenticator = Arc::new(XaiAuthenticator::new(
        tokens,
        XaiOAuth::new(auth_transport.clone()),
    ));
    let provider = provider_with(
        &mock,
        ProviderConfig::xai(Credentials::none()).with_authenticator(authenticator.clone()),
    );

    let error = provider
        .language_model("grok-4.5")
        .generate(text_request("hi"))
        .await
        .expect_err("dead refresh token should fail the request");
    assert_eq!(error.kind(), ErrorKind::Authentication);
    assert_eq!(authenticator.status().await, OAuthStatus::ReauthRequired);
    provider
        .language_model("grok-4.5")
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
    auth_transport.push_json(
        200,
        &json!({
            "access_token": "access-2",
            "refresh_token": "refresh-2",
            "expires_in": 3600,
        }),
    );

    let mut tokens = XaiTokens::new("access-1").with_refresh_token("refresh-1");
    tokens.expires_at = Some(now() - 10);
    let store = Arc::new(RecordingTokenStore::<XaiTokens>::failing());
    let authenticator = Arc::new(
        XaiAuthenticator::new(tokens, XaiOAuth::new(auth_transport)).with_token_store(store),
    );
    let provider = provider_with(
        &mock,
        ProviderConfig::xai(Credentials::none()).with_authenticator(authenticator.clone()),
    );

    let error = provider
        .language_model("grok-4.5")
        .generate(text_request("hi"))
        .await
        .expect_err("a failed durable save must fail authentication");

    assert_eq!(error.message(), "token store failed");
    assert_eq!(authenticator.tokens().await.access_token, "access-1");
    assert_eq!(authenticator.status().await, OAuthStatus::PersistenceFailed);
    assert!(mock.requests().is_empty());
}

#[tokio::test]
async fn unauthorized_recovery_retries_only_once() {
    let mock = MockTransport::shared();
    let unauthorized = json!({
        "error": {"message": "expired", "type": "authentication_error"}
    });
    mock.push_json(401, &unauthorized);
    mock.push_json(401, &unauthorized);

    let auth_transport = MockTransport::shared();
    auth_transport.push_json(
        200,
        &json!({
            "access_token": "access-2",
            "refresh_token": "refresh-2",
            "expires_in": 3600,
        }),
    );
    let mut tokens = XaiTokens::new("access-1").with_refresh_token("refresh-1");
    tokens.expires_at = Some(now() + 3600);
    let authenticator = XaiAuthenticator::new(tokens, XaiOAuth::new(auth_transport.clone()));
    let provider = provider_with(
        &mock,
        ProviderConfig::xai(Credentials::none()).with_authenticator(Arc::new(authenticator)),
    );

    let error = provider
        .language_model("grok-4.5")
        .generate(text_request("hi"))
        .await
        .expect_err("a second unauthorized response is terminal");

    assert_eq!(error.kind(), ErrorKind::Authentication);
    assert_eq!(mock.requests().len(), 2);
    assert_eq!(auth_transport.requests().len(), 1);
}

#[tokio::test]
async fn forbidden_responses_do_not_refresh_tokens() {
    let mock = MockTransport::shared();
    mock.push_json(
        403,
        &json!({"error": {"message": "no access to this model", "type": "permission_error"}}),
    );
    let auth_transport = MockTransport::shared();
    auth_transport.push_json(
        200,
        &json!({"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600}),
    );
    let mut tokens = XaiTokens::new("access-1").with_refresh_token("refresh-1");
    tokens.expires_at = Some(now() + 3600);
    let authenticator = XaiAuthenticator::new(tokens, XaiOAuth::new(auth_transport.clone()));
    let provider = provider_with(
        &mock,
        ProviderConfig::xai(Credentials::none()).with_authenticator(Arc::new(authenticator)),
    );

    let error = provider
        .language_model("grok-4.5")
        .generate(text_request("hi"))
        .await
        .expect_err("a forbidden response is not recoverable by refreshing");

    assert_eq!(error.kind(), ErrorKind::Permission);
    assert_eq!(mock.requests().len(), 1, "no retry");
    assert!(auth_transport.requests().is_empty(), "no refresh");
}

#[tokio::test]
async fn stale_unauthorized_request_reuses_the_new_token_generation() {
    let auth_transport = MockTransport::shared();
    auth_transport.push_json(
        200,
        &json!({
            "access_token": "access-2",
            "refresh_token": "refresh-2",
            "expires_in": 60,
        }),
    );
    let mut tokens = XaiTokens::new("access-1").with_refresh_token("refresh-1");
    tokens.expires_at = Some(now() + 3600);
    let authenticator = XaiAuthenticator::new(tokens, XaiOAuth::new(auth_transport.clone()));
    let url = Url::parse("https://api.x.ai/v1/responses").unwrap();
    let mut first = HttpRequest::post_json(url.clone(), &serde_json::json!({})).unwrap();
    let mut second = HttpRequest::post_json(url, &serde_json::json!({})).unwrap();
    authenticator.authenticate(&mut first).await.unwrap();
    authenticator.authenticate(&mut second).await.unwrap();

    let headers = caido_ai::transport::HeaderMap::new();
    let rejection = caido_ai::Rejection {
        status: 401,
        headers: &headers,
    };
    authenticator
        .reauthenticate(&mut first, &rejection)
        .await
        .unwrap();
    authenticator
        .reauthenticate(&mut second, &rejection)
        .await
        .unwrap();

    assert_eq!(header(&second, "authorization"), Some("Bearer access-2"));
    assert_eq!(auth_transport.requests().len(), 1);
}

#[tokio::test]
async fn expired_token_without_refresh_token_requires_sign_in() {
    let auth_transport = MockTransport::shared();
    let mut tokens = XaiTokens::new("expired-access");
    tokens.expires_at = Some(now() - 1);
    let authenticator = XaiAuthenticator::new(tokens, XaiOAuth::new(auth_transport.clone()));

    assert_eq!(authenticator.status().await, OAuthStatus::ReauthRequired);
    assert!(auth_transport.requests().is_empty());
}

#[tokio::test]
async fn failed_proactive_refresh_falls_back_to_the_valid_token() {
    let mock = MockTransport::shared();
    mock.push_json(200, &responses_completed("grok-4.5"));
    mock.push_json(200, &responses_completed("grok-4.5"));

    let auth_transport = MockTransport::shared();
    auth_transport.push_json(502, &json!({"error": "bad_gateway"}));

    // Inside the refresh skew but still valid for another minute.
    let mut tokens = XaiTokens::new("access-1").with_refresh_token("refresh-1");
    tokens.expires_at = Some(now() + 60);
    let authenticator = Arc::new(XaiAuthenticator::new(
        tokens,
        XaiOAuth::new(auth_transport.clone()),
    ));
    let provider = provider_with(
        &mock,
        ProviderConfig::xai(Credentials::none()).with_authenticator(authenticator.clone()),
    );

    // The first request tries and fails to refresh, and the second one is
    // inside the failure cooldown. Both must still go out with the valid token.
    for _ in 0..2 {
        provider
            .language_model("grok-4.5")
            .generate(text_request("hi"))
            .await
            .expect("a transient refresh failure must not fail the request");
    }

    assert_eq!(auth_transport.requests().len(), 1);
    for http in mock.requests() {
        assert_eq!(header(&http, "authorization"), Some("Bearer access-1"));
    }
    assert_eq!(authenticator.status().await, OAuthStatus::TransientFailure);
}
