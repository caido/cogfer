use super::*;

#[tokio::test]
async fn device_flow_start_poll_and_exchange() {
    let transport = MockTransport::shared();
    transport.push_json(
        200,
        &json!({"device_auth_id": "da_1", "usercode": "ABCD-EFGH", "interval": "3"}),
    );
    transport.push_json(403, &json!({"error": "authorization_pending"}));
    transport.push_json(
        200,
        &json!({"authorization_code": "code_1", "code_verifier": "verifier_1"}),
    );
    let access = jwt(now() + 3600, "acct_dev");
    transport.push_json(
        200,
        &json!({"access_token": access, "refresh_token": "refresh-dev"}),
    );

    let oauth = ChatGptOAuth::new(transport.clone()).with_client_config(
        cogfer::OAuthClientConfig::new("caido-client").expect("valid OAuth client"),
    );
    let device = oauth
        .start_device_authorization()
        .await
        .expect("device code issued");
    assert_eq!(device.user_code, "ABCD-EFGH");
    assert_eq!(
        device.verification_url,
        "https://auth.openai.com/codex/device"
    );
    assert_eq!(device.poll_interval, std::time::Duration::from_secs(3));

    assert!(matches!(
        oauth
            .poll_device_authorization(&device)
            .await
            .expect("pending poll is not an error"),
        DevicePoll::Pending
    ));

    let tokens = match oauth
        .poll_device_authorization(&device)
        .await
        .expect("confirmed poll succeeds")
    {
        DevicePoll::Complete(tokens) => tokens,
        other => panic!("expected Complete, got {other:?}"),
    };
    assert_eq!(tokens.access_token, access);
    assert_eq!(tokens.refresh_token.as_deref(), Some("refresh-dev"));
    assert_eq!(tokens.account_id.as_deref(), Some("acct_dev"));
    assert!(tokens.expires_at.is_some());

    let requests = transport.requests();
    assert_eq!(requests.len(), 4);
    assert_eq!(transport.request_json(0)["client_id"], "caido-client");
    assert_eq!(
        requests[0].url.as_str(),
        "https://auth.openai.com/api/accounts/deviceauth/usercode"
    );
    assert_eq!(
        requests[1].url.as_str(),
        "https://auth.openai.com/api/accounts/deviceauth/token"
    );
    let exchange = String::from_utf8(requests[3].body.clone().unwrap().to_vec()).unwrap();
    assert!(exchange.contains("grant_type=authorization_code"));
    assert!(exchange.contains("client_id=caido-client"));
    assert!(exchange.contains("code_verifier=verifier_1"));
    assert!(requests[3].headers.iter().any(
        |(name, value)| name == "content-type" && value == "application/x-www-form-urlencoded"
    ));
}

#[tokio::test]
async fn refresh_uses_the_configured_client_without_narrowing_scope() {
    let transport = MockTransport::shared();
    transport.push_json(
        200,
        &json!({"access_token": "access-2", "expires_in": 3600}),
    );
    let oauth = ChatGptOAuth::new(transport.clone()).with_client_config(
        cogfer::OAuthClientConfig::new("caido-client").expect("valid OAuth client"),
    );

    let tokens = oauth.refresh("refresh-1").await.expect("refresh succeeds");

    let form = String::from_utf8(transport.requests()[0].body.clone().unwrap().to_vec()).unwrap();
    assert_eq!(tokens.access_token, "access-2");
    assert!(form.contains("client_id=caido-client"));
    assert!(!form.contains("scope="));
}

#[tokio::test]
async fn device_flow_targets_the_configured_auth_base_url() {
    let transport = MockTransport::shared();
    transport.push_json(
        200,
        &json!({"device_auth_id": "da_1", "usercode": "ABCD-EFGH"}),
    );
    transport.push_json(
        200,
        &json!({"authorization_code": "code_1", "code_verifier": "verifier_1"}),
    );
    transport.push_json(200, &json!({"access_token": "opaque", "expires_in": 60}));
    transport.push_json(200, &json!({"access_token": "opaque-2", "expires_in": 60}));

    let oauth = ChatGptOAuth::new(transport.clone())
        .with_auth_base_url(Url::parse("http://localhost:5857/chatgpt-auth/").unwrap());
    let device = oauth
        .start_device_authorization()
        .await
        .expect("device code issued");
    assert_eq!(
        device.verification_url,
        "http://localhost:5857/chatgpt-auth/codex/device"
    );
    assert_eq!(device.expires_in, std::time::Duration::from_secs(15 * 60));
    assert!(matches!(
        oauth.poll_device_authorization(&device).await.unwrap(),
        DevicePoll::Complete(_)
    ));
    let refreshed = oauth.refresh("refresh-1").await.expect("refresh succeeds");
    assert_eq!(refreshed.access_token, "opaque-2");

    let urls: Vec<_> = transport
        .requests()
        .iter()
        .map(|request| request.url.to_string())
        .collect();
    assert_eq!(
        urls,
        [
            "http://localhost:5857/chatgpt-auth/api/accounts/deviceauth/usercode",
            "http://localhost:5857/chatgpt-auth/api/accounts/deviceauth/token",
            "http://localhost:5857/chatgpt-auth/oauth/token",
            "http://localhost:5857/chatgpt-auth/oauth/token",
        ]
    );
}
