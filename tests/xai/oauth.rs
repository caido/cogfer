use super::*;

#[tokio::test]
async fn device_flow_start_poll_and_tokens() {
    let transport = MockTransport::shared();
    transport.push_json(
        200,
        &json!({
            "device_code": "dc_1",
            "user_code": "ABCD-1234",
            "verification_uri": "https://accounts.x.ai/activate",
            "verification_uri_complete": "https://accounts.x.ai/activate?user_code=ABCD-1234",
            "interval": 3,
            "expires_in": 600,
        }),
    );
    transport.push_json(400, &json!({"error": "authorization_pending"}));
    transport.push_json(400, &json!({"error": "slow_down", "interval": 10}));
    transport.push_json(
        200,
        &json!({
            "access_token": "xai-access",
            "refresh_token": "xai-refresh",
            "expires_in": 3600,
            "token_type": "Bearer",
        }),
    );

    let oauth = XaiOAuth::new(transport.clone()).with_client_config(
        caido_ai::OAuthClientConfig::scoped("caido-client", "openid api:access")
            .expect("valid OAuth client"),
    );
    let device = oauth
        .start_device_authorization()
        .await
        .expect("device code issued");
    assert_eq!(device.user_code, "ABCD-1234");
    assert_eq!(device.verification_url, "https://accounts.x.ai/activate");
    assert_eq!(
        device.verification_url_complete.as_deref(),
        Some("https://accounts.x.ai/activate?user_code=ABCD-1234")
    );
    assert_eq!(device.poll_interval, Duration::from_secs(3));
    assert_eq!(device.expires_in, Duration::from_secs(600));

    assert!(matches!(
        oauth.poll_device_authorization(&device).await.unwrap(),
        DevicePoll::Pending
    ));
    match oauth.poll_device_authorization(&device).await.unwrap() {
        DevicePoll::SlowDown { interval } => {
            assert_eq!(interval, Some(Duration::from_secs(10)));
        }
        other => panic!("expected SlowDown, got {other:?}"),
    }
    let tokens = match oauth.poll_device_authorization(&device).await.unwrap() {
        DevicePoll::Complete(tokens) => tokens,
        other => panic!("expected Complete, got {other:?}"),
    };
    assert_eq!(tokens.access_token, "xai-access");
    assert_eq!(tokens.refresh_token.as_deref(), Some("xai-refresh"));
    let expires_at = tokens.expires_at.expect("expiry recorded");
    assert!((expires_at - now() - 3600).abs() <= 5);

    let requests = transport.requests();
    assert_eq!(requests.len(), 4);
    assert_eq!(
        requests[0].url.as_str(),
        "https://auth.x.ai/oauth2/device/code"
    );
    let start_form = form_body(&requests[0]);
    assert!(start_form.contains("client_id=caido-client"));
    assert!(start_form.contains("scope=openid+api%3Aaccess"));
    assert!(start_form.contains("referrer=caido-ai"));
    assert_eq!(requests[1].url.as_str(), "https://auth.x.ai/oauth2/token");
    let poll_form = form_body(&requests[1]);
    assert!(
        poll_form.contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code")
    );
    assert!(poll_form.contains("device_code=dc_1"));
    assert!(poll_form.contains("client_id=caido-client"));
}

#[tokio::test]
async fn denied_and_expired_device_codes_are_terminal() {
    let transport = MockTransport::shared();
    transport.push_json(
        200,
        &json!({
            "device_code": "dc_2",
            "user_code": "WXYZ-5678",
            "verification_uri": "https://accounts.x.ai/activate",
            "expires_in": 900,
        }),
    );
    transport.push_json(400, &json!({"error": "access_denied"}));
    transport.push_json(400, &json!({"error": "expired_token"}));

    let oauth = XaiOAuth::new(transport.clone());
    let device = oauth
        .start_device_authorization()
        .await
        .expect("device code issued");
    assert_eq!(device.poll_interval, Duration::from_secs(5));
    assert_eq!(device.expires_in, Duration::from_secs(15 * 60));

    let denied = oauth
        .poll_device_authorization(&device)
        .await
        .expect_err("denied codes end the flow");
    assert_eq!(denied.kind(), ErrorKind::Authentication);

    let expired = oauth
        .poll_device_authorization(&device)
        .await
        .expect_err("expired codes end the flow");
    assert_eq!(expired.kind(), ErrorKind::Timeout);
}

#[tokio::test]
async fn non_https_verification_uris_are_rejected() {
    let transport = MockTransport::shared();
    transport.push_json(
        200,
        &json!({
            "device_code": "dc_3",
            "user_code": "EVIL-0000",
            "verification_uri": "javascript:alert(1)",
            "expires_in": 900,
        }),
    );

    let error = XaiOAuth::new(transport)
        .start_device_authorization()
        .await
        .expect_err("non-https verification URIs are refused");
    assert_eq!(error.kind(), ErrorKind::MalformedResponse);
}

#[tokio::test]
async fn device_code_response_requires_an_expiry() {
    let transport = MockTransport::shared();
    transport.push_json(
        200,
        &json!({
            "device_code": "dc_4",
            "user_code": "ABCD-0000",
            "verification_uri": "https://accounts.x.ai/activate",
        }),
    );

    let error = XaiOAuth::new(transport)
        .start_device_authorization()
        .await
        .expect_err("RFC 8628 requires expires_in");

    assert_eq!(error.kind(), ErrorKind::MalformedResponse);
}

#[tokio::test]
async fn device_flow_targets_the_configured_auth_base_url() {
    let transport = MockTransport::shared();
    transport.push_json(
        200,
        &json!({
            "device_code": "dc_1",
            "user_code": "ABCD-1234",
            "verification_uri": "https://accounts.x.ai/activate",
            "expires_in": 600,
        }),
    );
    transport.push_json(
        200,
        &json!({"access_token": "xai-access", "expires_in": 3600}),
    );
    transport.push_json(
        200,
        &json!({"access_token": "xai-access-2", "expires_in": 3600}),
    );

    let oauth = XaiOAuth::new(transport.clone())
        .with_auth_base_url(Url::parse("http://localhost:5857/xai-auth").unwrap());
    let device = oauth
        .start_device_authorization()
        .await
        .expect("device code issued");
    assert!(matches!(
        oauth.poll_device_authorization(&device).await.unwrap(),
        DevicePoll::Complete(_)
    ));
    let refreshed = oauth.refresh("refresh-1").await.expect("refresh succeeds");
    assert_eq!(refreshed.access_token, "xai-access-2");

    let urls: Vec<_> = transport
        .requests()
        .iter()
        .map(|request| request.url.to_string())
        .collect();
    assert_eq!(
        urls,
        [
            "http://localhost:5857/xai-auth/oauth2/device/code",
            "http://localhost:5857/xai-auth/oauth2/token",
            "http://localhost:5857/xai-auth/oauth2/token",
        ]
    );
}
