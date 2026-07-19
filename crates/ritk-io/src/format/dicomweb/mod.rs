pub mod qido;
pub mod stow;
pub mod tests_dicomweb;
pub mod wado;

pub use qido::{QidoClient, QidoSearchParams};
pub use stow::{StowClient, StowFailure, StowResponse};
pub use wado::WadoClient;

/// Central DICOMweb HTTP client providing QIDO-RS, WADO-RS, and STOW-RS operations
/// against a single PACS base URL (PS 3.18).
///
/// URL prefix conventions used by this client:
/// - QIDO-RS:  `{base_url}/qido-rs/...`
/// - WADO-RS:  `{base_url}/wado-rs/...`
/// - STOW-RS:  `{base_url}/stow-rs/...`
///
/// Construct with [`DicomWebClient::new`] and optionally add an `Authorization` header
/// with [`DicomWebClient::with_auth`].
pub struct DicomWebClient {
    base_url: String,
    client: reqwest::blocking::Client,
    auth_header: Option<String>,
}

impl DicomWebClient {
    /// Constructs a new `DicomWebClient` targeting `base_url`.
    ///
    /// # Panics
    /// Panics only if `reqwest` cannot build a client from default configuration,
    /// which cannot occur under normal operating conditions.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: reqwest::blocking::ClientBuilder::new()
                .build()
                .expect("valid HTTP client config"),
            auth_header: None,
        }
    }

    /// Attaches a verbatim `Authorization` header value to every subsequent request.
    ///
    /// `auth` must be the full header value, e.g. `"Bearer <token>"` or `"Basic <base64>"`.
    pub fn with_auth(mut self, auth: impl Into<String>) -> Self {
        self.auth_header = Some(auth.into());
        self
    }

    // â”€â”€ QIDO-RS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Searches studies at `{base_url}/qido-rs/studies`.
    pub fn search_studies(
        &self,
        params: &QidoSearchParams,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let qido_base = format!("{}/qido-rs", self.base_url);
        QidoClient::new(&self.client, &qido_base, &self.auth_header).search("studies", params)
    }

    /// Searches series within a study at `{base_url}/qido-rs/studies/{study_uid}/series`.
    pub fn search_series(
        &self,
        study_uid: &str,
        params: &QidoSearchParams,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let qido_base = format!("{}/qido-rs", self.base_url);
        let endpoint = format!("studies/{}/series", study_uid);
        QidoClient::new(&self.client, &qido_base, &self.auth_header).search(&endpoint, params)
    }

    /// Searches instances within a series at
    /// `{base_url}/qido-rs/studies/{study_uid}/series/{series_uid}/instances`.
    pub fn search_instances(
        &self,
        study_uid: &str,
        series_uid: &str,
        params: &QidoSearchParams,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let qido_base = format!("{}/qido-rs", self.base_url);
        let endpoint = format!("studies/{}/series/{}/instances", study_uid, series_uid);
        QidoClient::new(&self.client, &qido_base, &self.auth_header).search(&endpoint, params)
    }

    // â”€â”€ WADO-RS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Retrieves a single DICOM P10 instance as raw bytes from
    /// `{base_url}/wado-rs/studies/{study_uid}/series/{series_uid}/instances/{sop_uid}`.
    pub fn retrieve_instance(
        &self,
        study_uid: &str,
        series_uid: &str,
        sop_uid: &str,
    ) -> anyhow::Result<Vec<u8>> {
        let wado_base = format!("{}/wado-rs", self.base_url);
        WadoClient::new(&self.client, &wado_base, &self.auth_header)
            .retrieve_instance(study_uid, series_uid, sop_uid)
    }

    // â”€â”€ STOW-RS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Stores DICOM P10 instances via a multipart/related POST to
    /// `{base_url}/stow-rs/studies[/{study_uid}]`.
    ///
    /// `parts` is a slice of `(filename, dicom_p10_bytes)` pairs.
    pub fn store_instances(
        &self,
        study_uid: Option<&str>,
        parts: &[(String, Vec<u8>)],
    ) -> anyhow::Result<StowResponse> {
        let stow_base = format!("{}/stow-rs", self.base_url);
        StowClient::new(&self.client, &stow_base, &self.auth_header).store(study_uid, parts)
    }
}
