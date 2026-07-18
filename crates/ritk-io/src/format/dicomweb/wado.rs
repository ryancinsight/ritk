use anyhow::Context;

/// Borrowed WADO-RS client that retrieves DICOM instances by reference.
pub struct WadoClient<'a> {
    client: &'a reqwest::blocking::Client,
    base_url: &'a str,
    auth_header: &'a Option<String>,
}

/// Constructs a WADO-RS instance retrieval URL.
///
/// Returns `{base}/studies/{study_uid}/series/{series_uid}/instances/{sop_uid}`
/// per PS 3.18 Â§10.4.
pub fn build_wado_url(base: &str, study_uid: &str, series_uid: &str, sop_uid: &str) -> String {
    format!(
        "{}/studies/{}/series/{}/instances/{}",
        base, study_uid, series_uid, sop_uid
    )
}

/// Issues an HTTP GET with `Accept: application/octet-stream` and returns the body bytes.
///
/// Used as the low-level transport primitive for WADO-RS instance retrieval.
pub fn retrieve_instance_bytes(
    client: &reqwest::blocking::Client,
    url: &str,
    auth: &Option<String>,
) -> anyhow::Result<Vec<u8>> {
    let mut req = client.get(url).header("Accept", "application/octet-stream");
    if let Some(a) = auth {
        req = req.header("Authorization", a);
    }
    let resp = req.send().context("WADO-RS GET request failed")?;
    let bytes = resp.bytes().context("WADO-RS response body read failed")?;
    Ok(bytes.to_vec())
}

impl<'a> WadoClient<'a> {
    /// Creates a new `WadoClient` borrowing the shared HTTP client and configuration.
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

    /// Retrieves a single DICOM P10 instance as raw bytes.
    pub fn retrieve_instance(
        &self,
        study_uid: &str,
        series_uid: &str,
        sop_uid: &str,
    ) -> anyhow::Result<Vec<u8>> {
        let url = build_wado_url(self.base_url, study_uid, series_uid, sop_uid);
        retrieve_instance_bytes(self.client, &url, self.auth_header)
    }
}
