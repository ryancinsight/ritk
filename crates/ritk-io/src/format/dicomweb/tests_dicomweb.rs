#[cfg(test)]
mod tests {
    use crate::format::dicomweb::qido::{build_qido_url, parse_qido_response, QidoSearchParams};
    use crate::format::dicomweb::stow::{
        build_multipart_body, build_stow_url, parse_stow_response, MULTIPART_BOUNDARY,
    };
    use crate::format::dicomweb::wado::build_wado_url;

    // 1. Studies endpoint with no params produces clean URL.
    #[test]
    fn build_qido_url_studies_no_params() {
        let url = build_qido_url(
            "http://pacs.local/qido-rs",
            "studies",
            &QidoSearchParams::default(),
        );
        assert_eq!(url, "http://pacs.local/qido-rs/studies");
    }

    // 2. PatientID param appears correctly in query string.
    #[test]
    fn build_qido_url_with_patient_id() {
        let params = QidoSearchParams {
            patient_id: Some("ABC123".into()),
            ..Default::default()
        };
        let url = build_qido_url("http://pacs.local/qido-rs", "studies", &params);
        assert_eq!(url, "http://pacs.local/qido-rs/studies?PatientID=ABC123");
    }

    // 3. Multiple params â€” StudyDate and Modality both appear separated correctly.
    #[test]
    fn build_qido_url_with_multiple_params() {
        let params = QidoSearchParams {
            study_date: Some("20230101".into()),
            modality: Some("CT".into()),
            ..Default::default()
        };
        let url = build_qido_url("http://pacs.local/qido-rs", "studies", &params);
        assert!(
            url.contains("StudyDate=20230101"),
            "StudyDate missing from URL: {}",
            url
        );
        assert!(
            url.contains("Modality=CT"),
            "Modality missing from URL: {}",
            url
        );
        assert!(url.contains('?'), "URL missing query separator: {}", url);
    }

    // 4. Limit and offset appear with correct values.
    #[test]
    fn build_qido_url_with_limit_offset() {
        let params = QidoSearchParams {
            limit: Some(10),
            offset: Some(5),
            ..Default::default()
        };
        let url = build_qido_url("http://pacs.local/qido-rs", "studies", &params);
        assert!(url.contains("limit=10"), "limit missing from URL: {}", url);
        assert!(url.contains("offset=5"), "offset missing from URL: {}", url);
    }

    // 5. WADO-RS instance URL contains correct path segments.
    #[test]
    fn build_wado_url_instance() {
        let url = build_wado_url(
            "http://pacs.local/wado-rs",
            "1.2.3.4",
            "5.6.7.8",
            "9.10.11.12",
        );
        assert_eq!(
            url,
            "http://pacs.local/wado-rs/studies/1.2.3.4/series/5.6.7.8/instances/9.10.11.12"
        );
    }

    // 6. STOW-RS URL without study UID.
    #[test]
    fn build_stow_url_no_study() {
        let url = build_stow_url("http://pacs.local/stow-rs", None);
        assert_eq!(url, "http://pacs.local/stow-rs/studies");
    }

    // 7. STOW-RS URL with study UID appended.
    #[test]
    fn build_stow_url_with_study() {
        let url = build_stow_url("http://pacs.local/stow-rs", Some("1.2.840.10008.5.1.4.1.1"));
        assert_eq!(
            url,
            "http://pacs.local/stow-rs/studies/1.2.840.10008.5.1.4.1.1"
        );
    }

    // 8. Single-part body contains boundary markers, MIME headers, and payload bytes.
    #[test]
    fn build_multipart_body_single_part() {
        let parts = vec![("instance.dcm".to_string(), b"DICOMDATA".to_vec())];
        let body = build_multipart_body(&parts, MULTIPART_BOUNDARY);
        let body_str = String::from_utf8_lossy(&body);
        assert!(
            body_str.contains(&format!("--{}", MULTIPART_BOUNDARY)),
            "opening boundary missing: {}",
            body_str
        );
        assert!(
            body_str.contains("Content-Type: application/dicom"),
            "MIME content-type missing: {}",
            body_str
        );
        assert!(
            body_str.contains("instance.dcm"),
            "filename missing: {}",
            body_str
        );
        assert!(
            body_str.contains("DICOMDATA"),
            "payload bytes missing: {}",
            body_str
        );
        assert!(
            body_str.contains(&format!("--{}--", MULTIPART_BOUNDARY)),
            "closing boundary missing: {}",
            body_str
        );
    }

    // 9. Two-part body contains both filenames, both payloads, and three boundary occurrences
    //    (one per part + one closing, where the closing is also prefixed by the open boundary).
    #[test]
    fn build_multipart_body_multiple_parts() {
        let parts = vec![
            ("a.dcm".to_string(), b"DATA_A".to_vec()),
            ("b.dcm".to_string(), b"DATA_B".to_vec()),
        ];
        let body = build_multipart_body(&parts, MULTIPART_BOUNDARY);
        let body_str = String::from_utf8_lossy(&body);

        assert!(body_str.contains("a.dcm"), "a.dcm missing: {}", body_str);
        assert!(body_str.contains("DATA_A"), "DATA_A missing: {}", body_str);
        assert!(body_str.contains("b.dcm"), "b.dcm missing: {}", body_str);
        assert!(body_str.contains("DATA_B"), "DATA_B missing: {}", body_str);

        // "--DICOMwebBoundary42" is a prefix of both part boundaries and the closing boundary,
        // so non-overlapping matches total: 2 part opens + 1 close = 3.
        let boundary_count = body_str
            .matches(&format!("--{}", MULTIPART_BOUNDARY))
            .count();
        assert_eq!(
            boundary_count, 3,
            "expected 2 part boundaries + 1 closing = 3 total, got {}: {}",
            boundary_count, body_str
        );
    }

    // 10. Empty JSON array parses to an empty Vec.
    #[test]
    fn parse_qido_response_empty_array() {
        let result = parse_qido_response(b"[]").unwrap();
        assert_eq!(result.len(), 0);
    }

    // 11. Single-entry JSON array returns one value containing the expected DICOM tag key.
    #[test]
    fn parse_qido_response_one_entry() {
        let json = br#"[{"00100020":{"vr":"LO","Value":["ABC123"]}}]"#;
        let result = parse_qido_response(json).unwrap();
        assert_eq!(result.len(), 1);
        assert!(
            result[0].get("00100020").is_some(),
            "tag 00100020 missing from parsed entry: {:?}",
            result[0]
        );
        let uid_val = result[0]["00100020"]["Value"][0]
            .as_str()
            .expect("Value[0] must be a string");
        assert_eq!(uid_val, "ABC123");
    }

    // 12. Empty body produces a StowResponse with no stored or failed entries.
    #[test]
    fn parse_stow_response_empty_body() {
        let result = parse_stow_response(b"").unwrap();
        assert_eq!(
            result.stored.len(),
            0,
            "stored should be empty for empty body"
        );
        assert_eq!(
            result.failed.len(),
            0,
            "failed should be empty for empty body"
        );
    }
}
