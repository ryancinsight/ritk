//! Native study navigator and multi-series loading tests.

use super::session;
use super::support::{control_center, series_card_center};
use crate::dicom::loader::{load_volume_from_series_info, scan_folder_for_series, tests::fixtures};
use crate::presentation::native_session::layout::{PanelGrid, WorkspaceLayout};
use crate::presentation::native_session::SeriesBrowser;
use metis_platform::native::{
    ModifierState, MouseButton, NativeApplication, NativeFlow, WindowEvent,
};

const SECOND_SERIES_UID: &str = "2.25.20260905002";
const THIRD_SERIES_UID: &str = "2.25.20260905004";
const FOURTH_SERIES_UID: &str = "2.25.20260905005";

/// A deterministic payload that no other fixture series produces.
///
/// Multi-series assertions compare each panel's exact pixels. When every
/// acquisition wrote identical samples, a panel rendering another series'
/// pixels still matched, so per-series cross-wiring was undetectable. Each
/// series is therefore given a payload derived from its `SeriesInstanceUID`,
/// and the derivation is a pure function of that UID so the expected values
/// can be reconstructed without re-reading the fixture.
pub(crate) fn distinct_samples(series_uid: &str) -> Vec<u8> {
    let stride = series_uid.bytes().fold(1_u32, |accumulator, byte| {
        accumulator.wrapping_mul(31).wrapping_add(u32::from(byte))
    }) | 1;
    fixtures::SAMPLES
        .iter()
        .enumerate()
        .map(|(index, &sample)| {
            let step = u32::try_from(index).unwrap_or(u32::MAX).wrapping_add(1);
            let stepped = u32::from(sample).wrapping_add(stride.wrapping_mul(step));
            u8::try_from(stepped & 0xFF).unwrap_or(u8::MAX)
        })
        .collect()
}

/// The stored payload each acquisition of a multi-series study encodes.
///
/// Every series is written from `distinct_samples`, so two panels can only
/// match while displaying the same acquisition. Identical payloads would let a
/// cross-wired panel pass an exact-pixel assertion.
fn acquisition_samples(series_uid: &str) -> Vec<u8> {
    distinct_samples(series_uid)
}

fn replacement_study() -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("replacement study root");
    let ct = acquisition_samples(fixtures::SERIES_UID);
    let mr = acquisition_samples(SECOND_SERIES_UID);
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID, &ct).expect("write CT series");
    fixtures::write_study(root.path(), "MR", SECOND_SERIES_UID, &mr).expect("write MR series");
    root
}

fn four_series_study() -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("four-series study root");
    let ct = acquisition_samples(fixtures::SERIES_UID);
    let mr = acquisition_samples(SECOND_SERIES_UID);
    let pt = acquisition_samples(THIRD_SERIES_UID);
    let us = acquisition_samples(FOURTH_SERIES_UID);
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID, &ct).expect("write CT series");
    fixtures::write_study(root.path(), "MR", SECOND_SERIES_UID, &mr).expect("write MR series");
    fixtures::write_study(root.path(), "PT", THIRD_SERIES_UID, &pt).expect("write PET series");
    fixtures::write_study(root.path(), "US", FOURTH_SERIES_UID, &us)
        .expect("write ultrasound series");
    root
}

/// Assert that panel `panel_index` displays exactly its assigned
/// acquisition's pixels.
///
/// The four-series study writes a distinct payload per `SeriesInstanceUID`, so
/// this fails if a panel renders another series' pixels — the cross-wiring the
/// previous `assert_ne!(BLACK)` check could not detect, because it accepted any
/// non-empty rendering.
fn assert_panel_displays_its_own_pixels(
    viewer: &crate::presentation::native_session::NativeViewerSession,
    expected_uids: &[&str],
) {
    let primary = expected_uids
        .first()
        .copied()
        .expect("at least the primary panel is expected");
    let panels = std::iter::once((&viewer.app, primary)).chain(
        viewer
            .compare_panels
            .iter()
            .zip(&expected_uids[1..])
            .map(|(panel, uid)| (&panel.app, *uid)),
    );
    for (panel_index, (app, uid)) in panels.enumerate() {
        let volume = app
            .loaded
            .as_ref()
            .unwrap_or_else(|| panic!("panel {panel_index} has no loaded volume"));
        let expected = fixtures::displayed_samples(&distinct_samples(uid));
        assert_eq!(
            volume.data.as_slice(),
            expected,
            "panel {panel_index} must display exactly series {uid}"
        );
        assert_eq!(
            volume
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.series_instance_uid.as_deref()),
            Some(uid),
            "panel {panel_index} metadata must name series {uid}"
        );
    }
}

/// Assert that panels showing distinct acquisitions render distinct images.
///
/// A panel that displays another series' pixels produces the same rendered
/// image as that series' own panel, so the rendered centers must be pairwise
/// distinct once the fixture payloads differ per series.
fn assert_panels_render_distinct_images(
    viewer: &crate::presentation::native_session::NativeViewerSession,
) {
    let centers: Vec<_> = viewer
        .viewports
        .iter()
        .map(|viewport| {
            let (x, y) = viewport.center();
            viewer.framebuffer.get_pixel(x, y)
        })
        .collect();
    for (index, center) in centers.iter().enumerate() {
        assert_ne!(
            *center,
            metis_platform::Color::BLACK,
            "panel {index} must display its own pixels"
        );
    }
    for (index, center) in centers.iter().enumerate() {
        assert!(
            centers[..index].iter().all(|other| other != center),
            "panel {index} must render a distinct image from every other panel"
        );
    }
}

fn click_series(
    session: &mut crate::presentation::native_session::NativeViewerSession,
    index: i32,
) {
    let (x, y) = series_card_center(session, usize::try_from(index).expect("series index"));
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x,
                y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x,
                y,
                button: MouseButton::Left,
            },
        ])
        .expect("select series from the study navigator");
}

fn drag_series_to_panel(
    session: &mut crate::presentation::native_session::NativeViewerSession,
    series_index: i32,
    panel_index: usize,
) {
    let (start_x, start_y) = series_card_center(
        session,
        usize::try_from(series_index).expect("series index"),
    );
    let (end_x, end_y) = session.viewports[panel_index].center();
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: start_x,
                y: start_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerMove { x: end_x, y: end_y },
            WindowEvent::PointerUp {
                x: end_x,
                y: end_y,
                button: MouseButton::Left,
            },
        ])
        .expect("drag the series card into the selected panel");
}

fn loaded_series_uid(app: &crate::app::SnapApp) -> Option<&str> {
    app.loaded
        .as_ref()?
        .metadata
        .as_ref()?
        .series_instance_uid
        .as_deref()
}

mod catalog;
mod multi_series_panels;
mod panel_input;
mod panel_lifecycle;
mod panel_routing;
mod panels;
