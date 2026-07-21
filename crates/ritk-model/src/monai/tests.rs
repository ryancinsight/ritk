//! Value-semantic tests for the MONAI Label Server client.
//!
//! Tests are grouped into three layers:
//! 1. Type serialisation/deserialisation (no I/O).
//! 2. Multipart response parsing (no I/O).
//! 3. HTTP client methods against a local `mockito` mock server.

use super::client::{parse_infer_response, MonaiLabelClient};
use super::types::{InferRequest, ModelInfo, ModelType, MonaiError, ServerInfo};

// ── Test helper ───────────────────────────────────────────────────────────────

/// Build a well-formed multipart/form-data body for test assertions.
///
/// `parts` = slice of `(name, content_type, body_bytes)`.
fn build_multipart(boundary: &str, parts: &[(&str, &str, &[u8])]) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, ct, data) in parts {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"{name}\"\r\nContent-Type: {ct}\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(data);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
}

// ── Layer 1 — Type serialisation ──────────────────────────────────────────────

/// Deserialise a full ServerInfo JSON; assert every field round-trips correctly.
///
/// Reference: MONAI Label Server GET /info response schema.
#[test]
fn test_server_info_deserialize_all_fields() {
    let json = r#"{"name":"monai","description":"MONAI Label","version":"0.5.0","labels":{"organ":"liver"}}"#;
    let info: ServerInfo = serde_json::from_str(json).expect("infallible: validated precondition");
    assert_eq!(info.name, "monai");
    assert_eq!(info.description, "MONAI Label");
    assert_eq!(info.version, "0.5.0");
    assert!(
        info.labels.is_object(),
        "labels must deserialise as a JSON object"
    );
}

/// ServerInfo with missing optional fields falls back to defaults.
#[test]
fn test_server_info_missing_optional_fields_default() {
    let json = r#"{"name":"minimal"}"#;
    let info: ServerInfo = serde_json::from_str(json).expect("infallible: validated precondition");
    assert_eq!(info.name, "minimal");
    assert_eq!(info.description, "");
    assert_eq!(info.version, "");
    assert!(info.labels.is_null());
}

/// ModelType::Segmentation round-trips through serde.
#[test]
fn test_model_type_segmentation_roundtrip() {
    let mt: ModelType =
        serde_json::from_str(r#""segmentation""#).expect("infallible: validated precondition");
    assert_eq!(mt, ModelType::Segmentation);
    let serialised = serde_json::to_string(&mt).expect("infallible: validated precondition");
    assert_eq!(serialised, r#""segmentation""#);
}

/// Unknown ModelType strings are preserved through serde without data loss.
#[test]
fn test_model_type_unknown_preserves_string() {
    let mt: ModelType =
        serde_json::from_str(r#""custom_net""#).expect("infallible: validated precondition");
    assert_eq!(mt, ModelType::Unknown("custom_net".to_owned()));
    let s = serde_json::to_string(&mt).expect("infallible: validated precondition");
    assert_eq!(s, r#""custom_net""#);
}

/// Deserialise a complete ModelInfo JSON object.
#[test]
fn test_model_info_deserialize_all_fields() {
    let json = r#"{"name":"seg_ct","description":"CT Seg","type":"segmentation","labels":["bg","liver","spleen"],"dimension":3}"#;
    let m: ModelInfo = serde_json::from_str(json).expect("infallible: validated precondition");
    assert_eq!(m.name, "seg_ct");
    assert_eq!(m.model_type, ModelType::Segmentation);
    assert_eq!(m.labels, vec!["bg", "liver", "spleen"]);
    assert_eq!(m.dimension, 3);
}

/// InferRequest::new produces a request with an empty JSON params object.
#[test]
fn test_infer_request_new_has_empty_params_object() {
    let req = InferRequest::new("seg_ct", "img-001");
    assert_eq!(req.model, "seg_ct");
    assert_eq!(req.image_id, "img-001");
    assert!(req.params.is_object(), "params must be a JSON object");
    assert_eq!(
        req.params
            .as_object()
            .expect("infallible: validated precondition")
            .len(),
        0,
        "default params object must be empty"
    );
}

// ── Layer 2 — Multipart response parsing ─────────────────────────────────────

/// parse_infer_response correctly extracts label bytes and JSON params.
///
/// Reference: MONAI Label Server POST /infer response format (multipart/form-data).
#[test]
fn test_parse_infer_response_label_and_params() {
    let label_bytes: &[u8] = &[0x01, 0x02, 0x03, 0x04, 0xAA, 0xBB];
    let params_json = br#"{"model":"seg","latencies":{"total":0.5}}"#;
    let boundary = "TestBoundary42";
    let ct = format!("multipart/form-data; boundary={boundary}");
    let body = build_multipart(
        boundary,
        &[
            ("label", "application/octet-stream", label_bytes),
            ("params", "application/json", params_json),
        ],
    );
    let resp = parse_infer_response(&ct, &body).expect("infallible: validated precondition");
    assert_eq!(resp.label, label_bytes, "label bytes must match exactly");
    assert_eq!(
        resp.params["model"]
            .as_str()
            .expect("infallible: validated precondition"),
        "seg",
        "model field must round-trip"
    );
    let total = resp.params["latencies"]["total"]
        .as_f64()
        .expect("infallible: validated precondition");
    assert!(
        (total - 0.5).abs() < 1e-9,
        "latency must be 0.5; got {total}"
    );
}

/// parse_infer_response returns ParseError when the label part is absent.
#[test]
fn test_parse_infer_response_missing_label_returns_parse_error() {
    let boundary = "B";
    let ct = format!("multipart/form-data; boundary={boundary}");
    let body = build_multipart(boundary, &[("params", "application/json", b"{}")]);
    let err = parse_infer_response(&ct, &body).unwrap_err();
    let MonaiError::ParseError { message } = err else {
        panic!("expected ParseError, got: {err:?}");
    };
    assert!(
        message.contains("label"),
        "error message must mention 'label'; got: {message}"
    );
}

/// parse_infer_response returns ParseError when Content-Type has no boundary.
#[test]
fn test_parse_infer_response_missing_boundary_returns_parse_error() {
    let err = parse_infer_response("application/json", b"{}").unwrap_err();
    assert!(
        matches!(err, MonaiError::ParseError { .. }),
        "expected ParseError for missing boundary"
    );
}

// ── Layer 3 — HTTP client via mockito ─────────────────────────────────────────

/// GET /info returns correct ServerInfo when the server responds with valid JSON.
#[test]
fn test_info_success() {
    let mut server = mockito::Server::new();
    let _m = server
        .mock("GET", "/info")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"name":"monai","description":"test","version":"0.5.0"}"#)
        .create();
    let client = MonaiLabelClient::new(server.url());
    let info = client.info().expect("infallible: validated precondition");
    assert_eq!(info.name, "monai");
    assert_eq!(info.version, "0.5.0");
}

/// GET /models returns a Vec<ModelInfo> with the name injected from the map key.
#[test]
fn test_models_success_name_injected_from_key() {
    let mut server = mockito::Server::new();
    let body = r#"{"seg_lung":{"type":"segmentation","labels":["bg","lung"],"dimension":3}}"#;
    let _m = server
        .mock("GET", "/models")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(body)
        .create();
    let client = MonaiLabelClient::new(server.url());
    let models = client.models().expect("infallible: validated precondition");
    assert_eq!(models.len(), 1, "must return exactly one model");
    let m = &models[0];
    assert_eq!(
        m.name, "seg_lung",
        "name must be injected from the JSON key"
    );
    assert_eq!(m.model_type, ModelType::Segmentation);
    assert_eq!(m.labels, vec!["bg", "lung"]);
    assert_eq!(m.dimension, 3);
}

/// GET /models propagates a server-side 500 error as MonaiError::ServerError.
#[test]
fn test_models_server_error_propagated() {
    let mut server = mockito::Server::new();
    let _m = server
        .mock("GET", "/models")
        .with_status(500)
        .with_body("Internal Server Error")
        .create();
    let client = MonaiLabelClient::new(server.url());
    let err = client.models().unwrap_err();
    let MonaiError::ServerError { status, body } = err else {
        panic!("expected ServerError, got: {err:?}");
    };
    assert_eq!(status, 500);
    assert!(
        body.contains("Internal Server Error"),
        "body must contain error text"
    );
}

/// POST /infer/{model}?image={id} returns InferResponse with correct label and params.
#[test]
fn test_infer_success_returns_label_and_params() {
    let mut server = mockito::Server::new();
    let label_bytes: Vec<u8> = b"NIFTI_MAGIC_BYTES".to_vec();
    let params_str = r#"{"model":"seg_ct","latencies":{"total":1.2}}"#;
    let boundary = "InferBoundary";
    let multipart_body = build_multipart(
        boundary,
        &[
            ("label", "application/octet-stream", &label_bytes),
            ("params", "application/json", params_str.as_bytes()),
        ],
    );
    let ct = format!("multipart/form-data; boundary={boundary}");
    let _m = server
        .mock("POST", "/infer/seg_ct?image=img-001")
        .with_status(200)
        .with_header("content-type", &ct)
        .with_body(multipart_body)
        .create();
    let client = MonaiLabelClient::new(server.url());
    let req = InferRequest::new("seg_ct", "img-001");
    let resp = client
        .infer(&req)
        .expect("infallible: validated precondition");
    assert_eq!(resp.label, label_bytes, "label bytes must match");
    assert_eq!(
        resp.params["model"]
            .as_str()
            .expect("infallible: validated precondition"),
        "seg_ct",
        "model name must round-trip"
    );
    let total = resp.params["latencies"]["total"]
        .as_f64()
        .expect("infallible: validated precondition");
    assert!(
        (total - 1.2).abs() < 1e-9,
        "latency must be 1.2; got {total}"
    );
}
