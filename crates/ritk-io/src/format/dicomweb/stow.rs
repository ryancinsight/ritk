use anyhow::Context;

/// Borrowed STOW-RS client that stores DICOM instances via multipart/related POST.
pub struct StowClient<'a> {
    client: &'a reqwest::blocking::Client,
    base_url: &'a str,
    auth_header: &'a Option<String>,
}

/// Outcome of a STOW-RS store operation (PS 3.18 Â§10.5).
#[derive(Debug, Clone)]
pub struct StowResponse {
    /// SOPInstanceUIDs of instances successfully stored.
    pub stored: Vec<String>,
    /// Instances that the SCP rejected, with DICOM failure reason codes.
    pub failed: Vec<StowFailure>,
}

/// A single STOW-RS failure entry from the response (PS 3.18 Â§10.5.1).
#[derive(Debug, Clone)]
pub struct StowFailure {
    pub sop_instance_uid: String,
    /// DICOM failure reason code (tag 00081197).
    pub failure_reason: u16,
}

/// Fixed MIME boundary used for deterministic body construction and test assertions.
pub const MULTIPART_BOUNDARY: &str = "DICOMwebBoundary42";

/// Constructs the STOW-RS target URL.
///
/// Returns `{base}/studies` or `{base}/studies/{study_uid}` when `study_uid` is `Some`,
/// per PS 3.18 Â§10.5.
pub fn build_stow_url(base: &str, study_uid: Option<&str>) -> String {
    match study_uid {
        None => format!("{}/studies", base),
        Some(uid) => format!("{}/studies/{}", base, uid),
    }
}

/// Constructs a `multipart/related` MIME body for STOW-RS per PS 3.18 Â§10.5.
///
/// Each element of `parts` contributes one MIME part with:
/// - `Content-Type: application/dicom`
/// - `Content-Disposition: form-data; name="{filename}"`
pub fn build_multipart_body(parts: &[(String, Vec<u8>)], boundary: &str) -> Vec<u8> {
    let mut body: Vec<u8> = Vec::new();
    for (filename, bytes) in parts {
        let header = format!(
            "--{}\r\nContent-Type: application/dicom\r\nContent-Disposition: form-data; name=\"{}\"\r\n\r\n",
            boundary, filename
        );
        body.extend_from_slice(header.as_bytes());
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    let closing = format!("--{}--\r\n", boundary);
    body.extend_from_slice(closing.as_bytes());
    body
}

/// Parses a STOW-RS JSON response body into a `StowResponse`.
///
/// Returns `StowResponse { stored: [], failed: [] }` on empty or non-JSON input,
/// satisfying the minimal contract for SCP implementations that return no body.
///
/// JSON field mapping (DICOM NativeDICOM JSON):
/// - `"00081199"`: ReferencedSOPSequence â€” successfully stored instances
/// - `"00081198"`: FailedSOPSequence â€” failed instances
/// - `"00081155"`: ReferencedSOPInstanceUID within each sequence item
/// - `"00081197"`: FailureReason code within each failed sequence item
pub fn parse_stow_response(body: &[u8]) -> anyhow::Result<StowResponse> {
    if body.is_empty() {
        return Ok(StowResponse {
            stored: vec![],
            failed: vec![],
        });
    }
    let Ok(val) = serde_json::from_slice::<serde_json::Value>(body) else {
        return Ok(StowResponse {
            stored: vec![],
            failed: vec![],
        });
    };
    let stored = extract_stored_uids(&val);
    let failed = extract_failed_instances(&val);
    Ok(StowResponse { stored, failed })
}

/// Extracts SOPInstanceUIDs from the `"00081199"` (ReferencedSOPSequence) tag.
fn extract_stored_uids(val: &serde_json::Value) -> Vec<String> {
    let Some(arr) = val
        .get("00081199")
        .and_then(|v| v.get("Value"))
        .and_then(|v| v.as_array())
    else {
        return vec![];
    };
    arr.iter()
        .filter_map(|item| {
            item.get("00081155")
                .and_then(|v| v.get("Value"))
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .collect()
}

/// Extracts failed instance entries from the `"00081198"` (FailedSOPSequence) tag.
fn extract_failed_instances(val: &serde_json::Value) -> Vec<StowFailure> {
    let Some(arr) = val
        .get("00081198")
        .and_then(|v| v.get("Value"))
        .and_then(|v| v.as_array())
    else {
        return vec![];
    };
    arr.iter()
        .filter_map(|item| {
            let uid = item
                .get("00081155")
                .and_then(|v| v.get("Value"))
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_str())
                .map(String::from)?;
            let reason = item
                .get("00081197")
                .and_then(|v| v.get("Value"))
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u16;
            Some(StowFailure {
                sop_instance_uid: uid,
                failure_reason: reason,
            })
        })
        .collect()
}

impl<'a> StowClient<'a> {
    /// Creates a new `StowClient` borrowing the shared HTTP client and configuration.
    pub fn new(
        client: &'a reqwest::blocking::Client,
        base_url: &'a str,
        auth_header: &'a Option<String>,
    ) -> Self {
        Self {
            client,
            base_url,
            auth_header,
        }
    }

    /// Stores DICOM P10 instances via STOW-RS multipart/related POST.
    ///
    /// `parts` is a slice of `(filename, dicom_bytes)` pairs. Each pair becomes
    /// one MIME part in the request body.
    pub fn store(
        &self,
        study_uid: Option<&str>,
        parts: &[(String, Vec<u8>)],
    ) -> anyhow::Result<StowResponse> {
        let url = build_stow_url(self.base_url, study_uid);
        let body = build_multipart_body(parts, MULTIPART_BOUNDARY);
        let content_type = format!(
            "multipart/related; type=\"application/dicom\"; boundary={}",
            MULTIPART_BOUNDARY
        );
        let mut req = self
            .client
            .post(&url)
            .header("Content-Type", &content_type)
            .body(body);
        if let Some(auth) = self.auth_header {
            req = req.header("Authorization", auth);
        }
        let resp = req.send().context("STOW-RS POST request failed")?;
        let resp_body = resp.bytes().context("STOW-RS response body read failed")?;
        parse_stow_response(&resp_body)
    }
}
