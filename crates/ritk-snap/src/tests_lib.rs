use super::*;

/// Verify `ModalityDisplay::for_modality` against the analytically derived
/// standard clinical window parameters documented in the struct-level rustdoc.
///
/// CT lung window: centre = -400, width = 1500 → range [-1150, 350] HU
/// MR brain window: centre = 600, width = 1200 → typical soft-tissue
/// US 8-bit range: centre = 128, width = 256 → [0, 255]
/// None / unknown: centre = 128, width = 256 → conservative default
#[test]
fn test_modality_display_ct_window_parameters() {
    let ct = ModalityDisplay::for_modality(Some("CT"));
    assert_eq!(
        ct.window_center, -400.0,
        "CT window_center must be -400.0 (standard lung window)"
    );
    assert_eq!(
        ct.window_width, 1500.0,
        "CT window_width must be 1500.0 (standard lung window)"
    );

    let mr = ModalityDisplay::for_modality(Some("MR"));
    assert_eq!(
        mr.window_center, 600.0,
        "MR window_center must be 600.0 (typical brain window)"
    );
    assert_eq!(
        mr.window_width, 1200.0,
        "MR window_width must be 1200.0 (typical brain window)"
    );

    let us = ModalityDisplay::for_modality(Some("US"));
    assert_eq!(
        us.window_center, 128.0,
        "US window_center must be 128.0 (8-bit acoustic range midpoint)"
    );
    assert_eq!(
        us.window_width, 256.0,
        "US window_width must be 256.0 (full 8-bit range)"
    );

    let unknown = ModalityDisplay::for_modality(None);
    assert_eq!(
        unknown.window_center, 128.0,
        "None modality must fall back to default centre 128.0"
    );
    assert_eq!(
        unknown.window_width, 256.0,
        "None modality must fall back to default width 256.0"
    );
}
