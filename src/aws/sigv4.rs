//! AWS Signature Version 4 over an [`HttpRequest`].
//!
//! Implements the header-based variant: a canonical request over the method,
//! double-encoded path, sorted query, and the signed headers `host`,
//! `content-type` (when present), `x-amz-date`, and `x-amz-security-token`
//! (for temporary credentials), hashed into a string to sign and signed with
//! the date/region/service-derived key.

use std::fmt::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

use ring::{digest, hmac};

use super::AwsCredentials;
use crate::error::{Error, Result};
use crate::http::{header_value, uri_encode};
use crate::transport::{HeaderName, HeaderValue, HttpRequest, header};

const ALGORITHM: &str = "AWS4-HMAC-SHA256";
const AMZ_DATE: HeaderName = HeaderName::from_static("x-amz-date");
const SECURITY_TOKEN: HeaderName = HeaderName::from_static("x-amz-security-token");

/// Sign `request` in place as of `now`, replacing any previous signature.
///
/// The body must be final: its hash is part of the signature. Signing sets
/// `host`, `x-amz-date`, `x-amz-security-token` for temporary credentials,
/// and `authorization`.
///
/// # Errors
///
/// Returns an error when the URL has no host or a credential is not a valid
/// header value.
pub fn sign_request(
    request: &mut HttpRequest,
    credentials: &AwsCredentials,
    region: &str,
    service: &str,
    now: SystemTime,
) -> Result<()> {
    let (date, timestamp) = amz_date(now);
    let host = request
        .url
        .host_str()
        .ok_or_else(|| Error::invalid_request("aws requests need a host to sign"))?;
    let host = match request.url.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_owned(),
    };

    request.headers.remove(header::AUTHORIZATION);
    request.headers.remove(SECURITY_TOKEN);
    request.headers.insert(header::HOST, header_value(&host)?);
    request.headers.insert(AMZ_DATE, header_value(&timestamp)?);
    if let Some(token) = &credentials.session_token {
        let mut value = header_value(token.expose())?;
        value.set_sensitive(true);
        request.headers.insert(SECURITY_TOKEN, value);
    }

    let mut signed: Vec<(String, String)> =
        [header::HOST, header::CONTENT_TYPE, AMZ_DATE, SECURITY_TOKEN]
            .iter()
            .filter_map(|name| {
                let value = request.headers.get(name)?.to_str().ok()?;
                Some((name.as_str().to_owned(), canonical_header_value(value)))
            })
            .collect();
    signed.sort();
    let signed_headers = signed
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>()
        .join(";");

    let mut canonical_request = String::new();
    canonical_request.push_str(request.method.as_str());
    canonical_request.push('\n');
    canonical_request.push_str(&canonical_uri(request.url.path()));
    canonical_request.push('\n');
    canonical_request.push_str(&canonical_query(&request.url));
    canonical_request.push('\n');
    for (name, value) in &signed {
        let _ = writeln!(canonical_request, "{name}:{value}");
    }
    canonical_request.push('\n');
    canonical_request.push_str(&signed_headers);
    canonical_request.push('\n');
    canonical_request.push_str(&sha256_hex(request.body.as_deref().unwrap_or_default()));

    let scope = format!("{date}/{region}/{service}/aws4_request");
    let string_to_sign = format!(
        "{ALGORITHM}\n{timestamp}\n{scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    );

    let mut key = hmac_sha256(
        format!("AWS4{}", credentials.secret_access_key.expose()).as_bytes(),
        date.as_bytes(),
    );
    for component in [region, service, "aws4_request"] {
        key = hmac_sha256(&key, component.as_bytes());
    }
    let signature = hex(&hmac_sha256(&key, string_to_sign.as_bytes()));

    let authorization = format!(
        "{ALGORITHM} Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
        credentials.access_key_id
    );
    let mut authorization: HeaderValue = header_value(&authorization)?;
    authorization.set_sensitive(true);
    request.headers.insert(header::AUTHORIZATION, authorization);
    Ok(())
}

/// The URI-encoded path, encoded once more per segment as AWS requires for
/// services other than S3.
fn canonical_uri(path: &str) -> String {
    if path.is_empty() {
        return "/".into();
    }
    path.split('/')
        .map(|segment| uri_encode(segment, true))
        .collect::<Vec<_>>()
        .join("/")
}

fn canonical_query(url: &url::Url) -> String {
    let mut pairs: Vec<(String, String)> = url
        .query_pairs()
        .map(|(key, value)| (uri_encode(&key, true), uri_encode(&value, true)))
        .collect();
    pairs.sort();
    pairs
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join("&")
}

/// Trimmed, with runs of spaces collapsed.
fn canonical_header_value(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn sha256_hex(data: &[u8]) -> String {
    hex(digest::digest(&digest::SHA256, data).as_ref())
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    hmac::sign(&hmac::Key::new(hmac::HMAC_SHA256, key), data)
        .as_ref()
        .to_vec()
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

/// The credential-scope date (`YYYYMMDD`) and the `x-amz-date` timestamp
/// (`YYYYMMDDTHHMMSSZ`) for `now`.
fn amz_date(now: SystemTime) -> (String, String) {
    let seconds = now
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or_default();
    let (year, month, day) = civil_from_days(seconds.div_euclid(86_400));
    let seconds_of_day = seconds.rem_euclid(86_400);
    let date = format!("{year:04}{month:02}{day:02}");
    let timestamp = format!(
        "{date}T{:02}{:02}{:02}Z",
        seconds_of_day / 3600,
        seconds_of_day % 3600 / 60,
        seconds_of_day % 60
    );
    (date, timestamp)
}

/// Proleptic Gregorian date for a day count since 1970-01-01.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_index + 2) / 5 + 1) as u32;
    let month = if month_index < 10 { month_index + 3 } else { month_index - 9 } as u32;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use url::Url;

    use super::*;
    use crate::transport::{HeaderMap, Method};

    /// Inverse of [`civil_from_days`], for building test timestamps.
    fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
        let year = if month <= 2 { year - 1 } else { year };
        let era = year.div_euclid(400);
        let year_of_era = year.rem_euclid(400);
        let month_index = if month > 2 { month - 3 } else { month + 9 } as i64;
        let day_of_year = (153 * month_index + 2) / 5 + i64::from(day) - 1;
        let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
        era * 146_097 + day_of_era - 719_468
    }

    fn at(year: i64, month: u32, day: u32, hour: u64, minute: u64, second: u64) -> SystemTime {
        let days = days_from_civil(year, month, day) as u64;
        UNIX_EPOCH + Duration::from_secs(days * 86_400 + hour * 3600 + minute * 60 + second)
    }

    /// The AWS SigV4 test-suite credentials and scope.
    fn suite_credentials() -> AwsCredentials {
        AwsCredentials::new("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
    }

    #[test]
    fn dates_round_trip_and_format() {
        for (year, month, day) in [(1970, 1, 1), (2000, 2, 29), (2015, 8, 30), (2100, 12, 31)] {
            assert_eq!(
                civil_from_days(days_from_civil(year, month, day)),
                (year, month, day)
            );
        }
        assert_eq!(
            amz_date(at(2015, 8, 30, 12, 36, 0)),
            ("20150830".to_string(), "20150830T123600Z".to_string())
        );
    }

    /// `get-vanilla` from the AWS SigV4 test suite.
    #[test]
    fn signs_the_get_vanilla_vector() {
        let mut request = HttpRequest {
            method: Method::GET,
            url: Url::parse("https://example.amazonaws.com/").unwrap(),
            headers: HeaderMap::new(),
            body: None,
        };

        sign_request(
            &mut request,
            &suite_credentials(),
            "us-east-1",
            "service",
            at(2015, 8, 30, 12, 36, 0),
        )
        .unwrap();

        assert_eq!(
            request.headers[header::AUTHORIZATION].to_str().unwrap(),
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=host;x-amz-date, \
             Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        );
        assert_eq!(request.headers[header::HOST], "example.amazonaws.com");
        assert_eq!(request.headers[AMZ_DATE], "20150830T123600Z");
    }

    /// `post-x-www-form-urlencoded` from the AWS SigV4 test suite.
    #[test]
    fn signs_the_post_form_vector() {
        let mut request = HttpRequest {
            method: Method::POST,
            url: Url::parse("https://example.amazonaws.com/").unwrap(),
            headers: HeaderMap::from_iter([(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/x-www-form-urlencoded"),
            )]),
            body: Some(bytes::Bytes::from_static(b"Param1=value1")),
        };

        sign_request(
            &mut request,
            &suite_credentials(),
            "us-east-1",
            "service",
            at(2015, 8, 30, 12, 36, 0),
        )
        .unwrap();

        assert_eq!(
            request.headers[header::AUTHORIZATION].to_str().unwrap(),
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=content-type;host;x-amz-date, \
             Signature=ff11897932ad3f4e8b18135d722051e5ac45fc38421b1da7b9d196a0fe09473a"
        );
    }

    #[test]
    fn session_tokens_are_signed_and_sent() {
        let credentials = suite_credentials().with_session_token("session-token");
        let mut request = HttpRequest {
            method: Method::POST,
            url: Url::parse("https://bedrock-runtime.eu-west-1.amazonaws.com/model/a%3Ab/invoke")
                .unwrap(),
            headers: HeaderMap::new(),
            body: Some(bytes::Bytes::from_static(b"{}")),
        };

        sign_request(
            &mut request,
            &credentials,
            "eu-west-1",
            "bedrock",
            at(2026, 1, 2, 3, 4, 5),
        )
        .unwrap();

        let authorization = request.headers[header::AUTHORIZATION].to_str().unwrap();
        assert!(
            authorization.contains("SignedHeaders=host;x-amz-date;x-amz-security-token,"),
            "{authorization}"
        );
        assert_eq!(request.headers[SECURITY_TOKEN], "session-token");
        assert!(request.headers[SECURITY_TOKEN].is_sensitive());
        assert!(request.headers[header::AUTHORIZATION].is_sensitive());
    }

    #[test]
    fn canonical_uri_encodes_each_segment_again() {
        assert_eq!(
            canonical_uri("/model/a%3Ab/invoke"),
            "/model/a%253Ab/invoke"
        );
        assert_eq!(canonical_uri(""), "/");
        assert_eq!(canonical_uri("/"), "/");
    }

    #[test]
    fn canonical_query_is_sorted_and_encoded() {
        let url = Url::parse("https://h/?b=2&a=x%20y&a=1").unwrap();
        assert_eq!(canonical_query(&url), "a=1&a=x%20y&b=2");
    }

    #[test]
    fn signing_again_replaces_the_previous_signature() {
        let mut request = HttpRequest {
            method: Method::GET,
            url: Url::parse("https://example.amazonaws.com/").unwrap(),
            headers: HeaderMap::new(),
            body: None,
        };
        let credentials = suite_credentials();
        sign_request(
            &mut request,
            &credentials,
            "us-east-1",
            "service",
            at(2015, 8, 30, 12, 36, 0),
        )
        .unwrap();
        let first = request.headers[header::AUTHORIZATION].clone();

        sign_request(
            &mut request,
            &credentials,
            "us-east-1",
            "service",
            at(2015, 8, 30, 12, 37, 0),
        )
        .unwrap();

        assert_ne!(request.headers[header::AUTHORIZATION], first);
        assert_eq!(
            request
                .headers
                .get_all(header::AUTHORIZATION)
                .iter()
                .count(),
            1
        );
        assert_eq!(request.headers[AMZ_DATE], "20150830T123700Z");
    }
}
