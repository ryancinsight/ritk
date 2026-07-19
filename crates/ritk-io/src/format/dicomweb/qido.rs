use anyhow::Context;

/// Search parameters for QIDO-RS queries (PS 3.18 §6.7).
///
/// Each field maps to a standard DICOM keyword query parameter.
/// `None` fields are omitted from the URL query string.
#[derive(Debug, Default, Clone)]
pub struct QidoSearchParams {
    pub patient_id: Option<String>,
    pub patient_name: Option<String>,
    pub study_date: Option<String>,
    pub modality: Option<String>,
    pub study_instance_uid: Option<String>,
    pub series_instance_uid: Option<String>,
    pub sop_instance_uid: Option<String>,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

/// Borrowed QIDO-RS client that issues queries against a single base URL.
pub struct QidoClient<'a> {
    client: &'a reqwest::blocking::Client,
    base_url: &'a str,
    auth_header: &'a Option<String>,
}

/// Constructs a QIDO-RS URL: `{base}/{endpoint}[?param=value&...]`.
///
/// Standard DICOM keyword names are used as query parameter names per PS 3.18 §8.3.4.
/// Parameters are appended only when `Some`; the first uses `?`, subsequent use `&`.
pub fn build_qido_url(base: &str, endpoint: &str, params: &QidoSearchParams) -> String {
    let mut query_parts: Vec<String> = Vec::new();

    if let Some(v) = &params.patient_id {
        query_parts.push(format!("PatientID={}", v));
    }
    if let Some(v) = &params.patient_name {
        query_parts.push(format!("PatientName={}", v));
    }
    if let Some(v) = &params.study_date {
        query_parts.push(format!("StudyDate={}", v));
    }
    if let Some(v) = &params.modality {
        query_parts.push(format!("Modality={}", v));
    }
    if let Some(v) = &params.study_instance_uid {
        query_parts.push(format!("StudyInstanceUID={}", v));
    }
    if let Some(v) = &params.series_instance_uid {
        query_parts.push(format!("SeriesInstanceUID={}", v));
    }
    if let Some(v) = &params.sop_instance_uid {
        query_parts.push(format!("SOPInstanceUID={}", v));
    }
    if let Some(v) = params.limit {
        query_parts.push(format!("limit={}", v));
    }
    if let Some(v) = params.offset {
        query_parts.push(format!("offset={}", v));
    }

    let path = format!("{}/{}", base, endpoint);
    if query_parts.is_empty() {
        path
    } else {
        format!("{}?{}", path, query_parts.join("&"))
    }
}

/// Parses a QIDO-RS JSON array response body into a `Vec` of JSON values.
pub fn parse_qido_response(body: &[u8]) -> anyhow::Result<Vec<serde_json::Value>> {
    serde_json::from_slice(body).context("failed to parse QIDO-RS JSON response")
}

impl<'a> QidoClient<'a> {
    /// Creates a new `QidoClient` borrowing the shared HTTP client and configuration.
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

    /// Issues a QIDO-RS search request for the given endpoint and parameters.
    ///
    /// `endpoint` examples: `"studies"`, `"studies/{uid}/series"`,
    /// `"studies/{uid}/series/{uid}/instances"`.
    pub fn search(
        &self,
        endpoint: &str,
        params: &QidoSearchParams,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let url = build_qido_url(self.base_url, endpoint, params);
        let mut req = self
            .client
            .get(&url)
            .header("Accept", "application/dicom+json");
        if let Some(auth) = self.auth_header {
            req = req.header("Authorization", auth);
        }
        let resp = req.send().context("QIDO-RS GET request failed")?;
        let body = resp.bytes().context("QIDO-RS response body read failed")?;
        parse_qido_response(&body)
    }
}
