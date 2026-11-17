//! Acceptance for off-thread study loading.
//!
//! Study discovery stays synchronous, but every selected series decodes on a
//! worker while the UI thread stays free. Panels publish independently as
//! their workers finish, cancellation drops every in-flight decode through
//! the generation guard, and the final pixels match a synchronous decode.

use super::series_browser::{assert_panel_displays_its_own_pixels, four_series_study};
use super::support::session;
use crate::dicom::loader::load_volume_from_series_info;
use crate::presentation::native_session::NativeViewerSession;
use metis_platform::native::{ModifierState, NativeApplication, WindowEvent};

fn open_two_series(viewer: &mut NativeViewerSession) -> (tempfile::TempDir, Vec<String>) {
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    let uids = [0, 1]
        .iter()
        .map(|&index| {
            viewer
                .series_browser
                .as_ref()
                .expect("study catalog remains available")
                .choice(index)
                .expect("fixture series is catalogued")
                .acquisition
                .series_instance_uid()
                .to_owned()
        })
        .collect();
    viewer
        .open_selected_series(&[0, 1])
        .expect("queue two series without blocking");
    (study, uids)
}

/// Nothing decodes on the calling thread: right after the open call every
/// panel is still empty while the progress line reports the pending work.
#[test]
fn opening_series_does_not_block_on_decoding() {
    let (mut viewer, _initial_root) = session();
    let (_study, _uids) = open_two_series(&mut viewer);

    assert!(viewer.app.loaded.is_none());
    assert!(viewer
        .compare_panels
        .iter()
        .all(|panel| panel.app.loaded.is_none()));
    assert_eq!(viewer.series_load_total, 2);
    assert_eq!(viewer.app.status_message, "Loading series 0/2\u{2026}");
}

/// Joining only the primary worker publishes the first panel while the
/// second panel stays unpublished, even if its worker already finished
/// decoding: publication happens only through an explicit poll or join.
#[test]
fn panels_publish_independently_as_their_workers_finish() {
    let (mut viewer, _study_root) = session();
    let (_study, uids) = open_two_series(&mut viewer);

    viewer.app.wait_for_load_tasks();

    let primary = viewer.app.loaded.as_ref().expect("primary panel published");
    assert_eq!(
        primary
            .metadata
            .as_ref()
            .expect("primary metadata")
            .series_instance_uid
            .as_deref(),
        Some(uids[0].as_str()),
        "primary panel shows the first selected series"
    );
    assert!(
        viewer.compare_panels[0].app.loaded.is_none(),
        "second panel stays unpublished until it is polled"
    );

    viewer.wait_for_series_loads();
    viewer.refresh_frame().expect("render loaded series");
    assert_eq!(
        viewer.compare_panels[0]
            .app
            .loaded
            .as_ref()
            .expect("second panel published")
            .metadata
            .as_ref()
            .expect("second panel metadata")
            .series_instance_uid
            .as_deref(),
        Some(uids[1].as_str()),
        "second panel publishes the second series"
    );
    assert_panel_displays_its_own_pixels(&viewer, &[uids[0].as_str(), uids[1].as_str()]);
}

/// The asynchronously published pixels match a synchronous decode exactly.
#[test]
fn async_load_matches_synchronous_decode_pixel_for_pixel() {
    let (mut viewer, _study_root) = session();
    let (_study, uids) = open_two_series(&mut viewer);
    let reference: Vec<_> = [0, 1]
        .iter()
        .map(|&index| {
            let acquisition = viewer
                .series_browser
                .as_ref()
                .expect("study catalog remains available")
                .choice(index)
                .expect("fixture series is catalogued")
                .acquisition
                .clone();
            (
                uids[index].clone(),
                load_volume_from_series_info(&acquisition).expect("synchronous reference decode"),
            )
        })
        .collect();

    viewer.wait_for_series_loads();

    let panels =
        std::iter::once(&viewer.app).chain(viewer.compare_panels.iter().map(|panel| &panel.app));
    for ((uid, expected), app) in reference.iter().zip(panels) {
        let actual = app.loaded.as_ref().expect("panel published");
        assert_eq!(
            actual.data.as_slice(),
            expected.data.as_slice(),
            "series {uid} async pixels match the synchronous decode"
        );
    }
}

/// Cancelling clears every in-flight decode, so a later drain publishes
/// nothing; generations are bumped, which also discards workers that
/// already finished decoding but were never polled.
#[test]
fn cancelling_a_study_load_publishes_nothing() {
    let (mut viewer, _study_root) = session();
    let (_study, _uids) = open_two_series(&mut viewer);

    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x1b,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("cancel loading with Escape");
    assert_eq!(viewer.app.status_message, "Series loading cancelled.");

    viewer.wait_for_series_loads();
    viewer.refresh_frame().expect("render empty panels");

    assert!(viewer.app.loaded.is_none());
    assert!(viewer
        .compare_panels
        .iter()
        .all(|panel| panel.app.loaded.is_none()));
    assert!(!viewer.cancel_series_loads(), "second cancel is a no-op");
}
