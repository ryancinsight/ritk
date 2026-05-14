//! Colormap auto-selection tests by modality.

use super::*;

#[test]
fn colormap_for_modality_pt_yields_hot() {
    assert_eq!(
        SnapApp::colormap_for_modality(Some("PT")),
        Colormap::Hot,
        "PT modality must auto-select Hot colormap"
    );
}

#[test]
fn colormap_for_modality_ct_yields_grayscale() {
    assert_eq!(
        SnapApp::colormap_for_modality(Some("CT")),
        Colormap::Grayscale,
        "CT modality must auto-select Grayscale colormap"
    );
}

#[test]
fn colormap_for_modality_none_yields_grayscale() {
    assert_eq!(
        SnapApp::colormap_for_modality(None),
        Colormap::Grayscale,
        "absent modality must default to Grayscale colormap"
    );
}

#[test]
fn secondary_colormap_auto_selects_hot_when_secondary_is_pt() {
    let mut app = SnapApp::default();
    assert_eq!(app.secondary_colormap, Colormap::Grayscale);
    app.secondary_colormap = SnapApp::colormap_for_modality(Some("PT"));
    assert_eq!(
        app.secondary_colormap,
        Colormap::Hot,
        "secondary PT volume must produce Hot secondary colormap"
    );
}

#[test]
fn secondary_colormap_remains_grayscale_when_secondary_is_ct() {
    let mut app = SnapApp::default();
    app.secondary_colormap = SnapApp::colormap_for_modality(Some("CT"));
    assert_eq!(
        app.secondary_colormap,
        Colormap::Grayscale,
        "secondary CT volume must retain Grayscale secondary colormap"
    );
}

#[test]
fn primary_colormap_auto_selects_hot_when_primary_is_pt() {
    let mut app = SnapApp::default();
    assert_eq!(app.colormap, Colormap::Grayscale);
    app.colormap = SnapApp::colormap_for_modality(Some("PT"));
    assert_eq!(
        app.colormap,
        Colormap::Hot,
        "primary PT volume must produce Hot primary colormap"
    );
}
