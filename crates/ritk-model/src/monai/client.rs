//! MONAI Label Server REST client implementation.
//!
//! # Protocol
//!
//! Implements three endpoints of the MONAI Label Server REST API
//! (<https://docs.monai.io/projects/label/en/latest/apis.html>):
//!
//! | Method | Path | Description |
//! |---|---|---|
//! | GET | `/info` | Server metadata |
//! | GET | `/models` | Available models |
//! | POST | `/infer/{model}?image={id}` | Run segmentation inference |
//!
//! The client is **synchronous** (`reqwest::blocking`), preserving the async-contagion
//! prohibition: pure domain callers remain sync; the network I/O layer is isolated here.

use super::multipart::{extract_part_name, split_multipart};
use super::types::{InferRequest, InferResponse, ModelInfo, MonaiError, ServerInfo};
use reqwest::blocking::Client;
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, instrument};

// â”€â”€ MonaiLabelClient â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Synchronous REST client for MONAI Label Server inference endpoints.
///
/// # Thread safety
///
/// `MonaiLabelClient` is `Send + Sync` because `reqwest::blocking::Client` is
/// `Send + Sync`.  Multiple threads may share a reference to a single client.
pub struct MonaiLabelClient {
    base_url: String,
    client: Client,
}

impl MonaiLabelClient {
    /// Create a client with a 30-second connect/read timeout.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::with_timeout(base_url, Duration::from_secs(30))
    }

    /// Create a client with an explicit connect/read timeout.
    ///
    /// # Panics
    /// Panics only if the underlying TLS configuration is invalid, which cannot
    /// occur under normal operating conditions.
    pub fn with_timeout(base_url: impl Into<String>, timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("reqwest blocking Client must build");
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            client,
        }
    }

    // â”€â”€ GET /info â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Retrieve server metadata from `GET /info`.
    ///
    /// Returns [`ServerInfo`] on success or a [`MonaiError`] on any failure.
    #[instrument(skip(self), fields(base_url = %self.base_url))]
    pub fn info(&self) -> Result<ServerInfo, MonaiError> {
        let url = format!("{}/info", self.base_url);
        debug!("GET {url}");
        let resp = self.client.get(&url).send()?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            return Err(MonaiError::ServerError {
                status: status.as_u16(),
                body,
            });
        }
        let info: ServerInfo = resp.json()?;
        Ok(info)
    }

    // â”€â”€ GET /models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// List available inference models from `GET /models`.
    ///
    /// MONAI returns a JSON object mapping model-name â†’ model-metadata.  The model
    /// `name` field is injected from the map key when absent in the value object.
    #[instrument(skip(self), fields(base_url = %self.base_url))]
    pub fn models(&self) -> Result<Vec<ModelInfo>, MonaiError> {
        let url = format!("{}/models", self.base_url);
        debug!("GET {url}");
        let resp = self.client.get(&url).send()?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            return Err(MonaiError::ServerError {
                status: status.as_u16(),
                body,
            });
        }
        let raw: HashMap<String, serde_json::Value> = resp.json()?;
        raw.into_iter()
            .map(|(key, mut val)| {
                // Inject the map key as "name" when the value object lacks the field.
                if val.get("name").is_none() {
                    if let Some(obj) = val.as_object_mut() {
                        obj.insert("name".to_owned(), serde_json::Value::String(key));
                    }
                }
                serde_json::from_value::<ModelInfo>(val).map_err(MonaiError::Json)
            })
            .collect()
    }

    // â”€â”€ POST /infer/{model} â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Run AI inference via `POST /infer/{model}?image={image_id}`.
    ///
    /// `request.params` is serialised as JSON in the request body.  The server
    /// responds with `multipart/form-data` containing:
    /// - `label`  â€” binary NIfTI segmentation mask.
    /// - `params` â€” JSON inference metadata (timing, confidence, model version).
    ///
    /// # Errors
    ///
    /// Returns [`MonaiError::ParseError`] if the `label` part is absent in the response.
    #[instrument(skip(self, request), fields(model = %request.model, image_id = %request.image_id))]
    pub fn infer(&self, request: &InferRequest) -> Result<InferResponse, MonaiError> {
        let url = format!(
            "{}/infer/{}?image={}",
            self.base_url, request.model, request.image_id
        );
        debug!("POST {url}");
        let resp = self.client.post(&url).json(&request.params).send()?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            return Err(MonaiError::ServerError {
                status: status.as_u16(),
                body,
            });
        }
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_owned();
        let body_bytes = resp.bytes()?.to_vec();
        parse_infer_response(&content_type, &body_bytes)
    }
}

// â”€â”€ Response parser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Parse a MONAI Label Server multipart infer response into [`InferResponse`].
///
/// # Format
///
/// ```text
/// Content-Type: multipart/form-data; boundary=<B>
///
/// --<B>
/// Content-Disposition: form-data; name="label"; filename="label.nii.gz"
/// Content-Type: application/octet-stream
///
/// <binary NIfTI data>
/// --<B>
/// Content-Disposition: form-data; name="params"
/// Content-Type: application/json
///
/// {"model": ..., "latencies": {...}}
/// --<B>--
/// ```
///
/// # Errors
///
/// Returns [`MonaiError::ParseError`] if no multipart boundary is found in
/// `content_type` or if the `label` part is absent in the response.
pub(crate) fn parse_infer_response(
    content_type: &str,
    body: &[u8],
) -> Result<InferResponse, MonaiError> {
    let boundary = extract_boundary(content_type).ok_or_else(|| MonaiError::ParseError {
        message: format!("missing multipart boundary in Content-Type: {content_type}"),
    })?;

    let parts = split_multipart(body, boundary.as_bytes());
    let mut label: Option<Vec<u8>> = None;
    let mut params = serde_json::Value::Object(Default::default());

    for (hdr, body_part) in &parts {
        match extract_part_name(hdr).as_deref() {
            Some("label") => {
                label = Some(body_part.to_vec());
            }
            Some("params") => {
                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(body_part) {
                    params = v;
                }
            }
            _ => {}
        }
    }

    let label = label.ok_or_else(|| MonaiError::ParseError {
        message: "missing 'label' part in multipart infer response".to_owned(),
    })?;

    Ok(InferResponse { label, params })
}

/// Extract the `boundary` parameter value from a `multipart/form-data` Content-Type header.
///
/// Handles both quoted (`boundary="B"`) and unquoted (`boundary=B`) forms.
/// Returns `None` if the `boundary=` directive is absent.
fn extract_boundary(content_type: &str) -> Option<String> {
    content_type.split(';').find_map(|part| {
        let p = part.trim();
        p.strip_prefix("boundary=")
            .map(|b| b.trim_matches('"').to_owned())
    })
}
