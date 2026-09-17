use cogfer::transport::mock::MockTransport;
use cogfer::{Credentials, ProviderConfig};
use serde_json::json;
use url::Url;

use crate::common::{provider_with, text_request};

#[tokio::test]
async fn custom_gemini_base_preserves_prefix() {
    let mock = MockTransport::shared();
    mock.push_json(
        200,
        &json!({"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]},
                                 "finishReason": "STOP", "index": 0}]}),
    );
    let provider = provider_with(
        &mock,
        ProviderConfig::gemini(Credentials::api_key("k"))
            .with_base_url(Url::parse("http://localhost:8080/gemini/v1beta/").unwrap()),
    );
    provider
        .language_model("gemini-2.5-flash")
        .generate(text_request("hi"))
        .await
        .unwrap();
    assert_eq!(
        mock.requests()[0].url.as_str(),
        "http://localhost:8080/gemini/v1beta/models/gemini-2.5-flash:generateContent"
    );
}

#[tokio::test]
async fn streaming_custom_gemini_base_preserves_query_parameters() {
    let mock = MockTransport::shared();
    mock.push_sse(&[
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP","index":0}],"responseId":"r1"}"#,
    ]);
    let provider = provider_with(
        &mock,
        ProviderConfig::gemini(Credentials::api_key("k")).with_base_url(
            Url::parse("http://localhost:8080/gemini/v1beta/?project=caido").unwrap(),
        ),
    );

    let _stream = provider
        .language_model("gemini-3-flash")
        .stream(text_request("hi"))
        .await
        .unwrap();

    assert_eq!(
        mock.requests()[0].url.as_str(),
        "http://localhost:8080/gemini/v1beta/models/gemini-3-flash:streamGenerateContent?project=caido&alt=sse"
    );
}
