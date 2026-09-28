//! Native study navigator and multi-series loading tests.

use super::session;
use crate::dicom::loader::{load_volume_from_series_info, scan_folder_for_series, tests::fixtures};
use crate::presentation::native_session::layout::{PanelGrid, WorkspaceLayout};
use crate::presentation::native_session::SeriesBrowser;
use metis_platform::native::{
    ModifierState, MouseButton, NativeApplication, NativeFlow, WindowEvent,
};

const SECOND_SERIES_UID: &str = "2.25.20260905002";
const THIRD_SERIES_UID: &str = "2.25.20260905004";
const FOURTH_SERIES_UID: &str = "2.25.20260905005";

fn replacement_study() -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("replacement study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write CT series");
    fixtures::write_study(root.path(), "MR", SECOND_SERIES_UID).expect("write MR series");
    root
}

fn four_series_study() -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("four-series study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write CT series");
    fixtures::write_study(root.path(), "MR", SECOND_SERIES_UID).expect("write MR series");
    fixtures::write_study(root.path(), "PT", THIRD_SERIES_UID).expect("write PET series");
    fixtures::write_study(root.path(), "US", FOURTH_SERIES_UID).expect("write ultrasound series");
    root
}

fn click_series(
    session: &mut crate::presentation::native_session::NativeViewerSession,
    index: i32,
) {
    let row = index;
    let x = 24;
    let y = 190 + row.saturating_mul(95);
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
    let row = series_index;
    let start_x = 24;
    let start_y = 190 + row.saturating_mul(95);
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

#[test]
fn native_study_navigator_retains_and_loads_each_discovered_series() {
    let (mut viewer, _initial_root) = session();
    let replacement = replacement_study();
    viewer
        .open_study_path(replacement.path())
        .expect("open replacement study");

    let browser = viewer.series_browser.as_ref().expect("study navigator");
    assert_eq!(browser.len(), 2);
    assert_eq!(browser.active_index(), 0);
    assert_eq!(
        viewer
            .app
            .loaded
            .as_ref()
            .expect("active volume")
            .modality
            .as_deref(),
        Some("CT")
    );

    click_series(&mut viewer, 1);

    assert_eq!(
        viewer
            .app
            .loaded
            .as_ref()
            .expect("selected volume")
            .metadata
            .as_ref()
            .expect("selected metadata")
            .series_instance_uid
            .as_deref(),
        Some(SECOND_SERIES_UID)
    );
    let browser = viewer.series_browser.as_ref().expect("retained navigator");
    assert_eq!(browser.len(), 2);
    assert_eq!(browser.active_index(), 1);
}

#[test]
fn failed_series_switch_preserves_the_active_volume_and_catalog() {
    let (mut viewer, _initial_root) = session();
    let replacement = replacement_study();
    viewer
        .open_study_path(replacement.path())
        .expect("open replacement study");
    let previous_volume = viewer.app.loaded.as_ref().expect("active volume");
    let previous_uid = previous_volume
        .metadata
        .as_ref()
        .expect("active metadata")
        .series_instance_uid;
    let previous_shape = previous_volume.shape;
    let missing_paths = viewer
        .series_browser
        .as_ref()
        .expect("study navigator")
        .choice(1)
        .expect("second series")
        .acquisition
        .file_paths
        .clone();
    for path in missing_paths {
        std::fs::remove_file(path).expect("remove selected series instance");
    }

    click_series(&mut viewer, 1);

    assert_eq!(
        viewer.app.loaded.as_ref().expect("active volume").shape,
        previous_shape
    );
    assert_eq!(
        viewer
            .app
            .loaded
            .as_ref()
            .expect("active volume")
            .metadata
            .as_ref()
            .expect("active metadata")
            .series_instance_uid,
        previous_uid
    );
    assert_eq!(
        viewer
            .series_browser
            .as_ref()
            .expect("retained navigator")
            .active_index(),
        0
    );
    assert!(viewer
        .app
        .status_message
        .contains("Series could not be opened"));
}

#[test]
fn study_browser_input_does_not_start_a_viewport_gesture() {
    let (mut viewer, _initial_root) = session();
    let flow = viewer
        .handle_events(&[WindowEvent::PointerDown {
            x: 16,
            y: 700,
            button: MouseButton::Left,
        }])
        .expect("navigator input");

    assert_eq!(flow, NativeFlow::Continue { repaint: false });
    assert_eq!(viewer.active_view, None);
}

#[test]
fn comparison_displays_two_independent_series_and_routes_each_panel() {
    let (mut viewer, _initial_root) = session();
    let replacement = replacement_study();
    viewer
        .open_study_path(replacement.path())
        .expect("open two-series study");
    viewer.refresh_frame().expect("render first series");

    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: 650,
                y: 65,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: 650,
                y: 65,
                button: MouseButton::Left,
            },
            WindowEvent::PointerDown {
                x: 632,
                y: 151,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: 632,
                y: 151,
                button: MouseButton::Left,
            },
        ])
        .expect("enable side-by-side comparison");
    assert!(viewer.workspace_layout.is_grid());
    assert_eq!(viewer.active_panel, 1);
    click_series(&mut viewer, 1);

    {
        let primary = viewer
            .app
            .loaded
            .as_ref()
            .expect("primary series remains loaded");
        let secondary = viewer
            .compare_panels
            .first()
            .expect("comparison panel exists")
            .app
            .loaded
            .as_ref()
            .expect("second series is loaded");
        let primary_uid = primary
            .metadata
            .as_ref()
            .expect("primary metadata")
            .series_instance_uid
            .as_deref();
        let secondary_uid = secondary
            .metadata
            .as_ref()
            .expect("secondary metadata")
            .series_instance_uid
            .as_deref();
        assert_eq!(primary_uid, Some(fixtures::SERIES_UID));
        assert_eq!(secondary_uid, Some(SECOND_SERIES_UID));
        assert_ne!(primary_uid, secondary_uid);
    }
    assert!(viewer.viewports[0].contains(300.0, 300.0));
    assert!(viewer.viewports[1].contains(900.0, 300.0));
    for index in 0..2 {
        let (x, y) = viewer.viewports[index].center();
        assert_ne!(
            viewer.framebuffer.get_pixel(x, y),
            metis_platform::Color::BLACK,
            "comparison panel {index} must display its loaded series"
        );
    }

    let primary_slice = viewer.app.viewer_state.slice_index;
    let secondary_slice = viewer
        .compare_panels
        .first()
        .expect("comparison panel exists")
        .app
        .viewer_state
        .slice_index;
    let (right_x, right_y) = viewer.viewports[1].center();
    viewer
        .handle_events(&[WindowEvent::PointerWheel {
            x: right_x,
            y: right_y,
            delta_x: 0,
            delta_y: -120,
            modifiers: ModifierState::NONE,
        }])
        .expect("advance only the comparison panel");
    assert_eq!(viewer.app.viewer_state.slice_index, primary_slice);
    assert_ne!(
        viewer
            .compare_panels
            .first()
            .expect("comparison panel")
            .app
            .viewer_state
            .slice_index,
        secondary_slice
    );
}

#[test]
fn four_panel_layout_assigns_four_independent_dicom_series() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 2).expect("four-panel grid is within the picker bounds"),
        ))
        .expect("select four-panel layout");
    viewer.refresh_frame().expect("render four panel targets");

    click_series(&mut viewer, 1);
    for (panel_index, series_index) in [(2, 2), (3, 3)] {
        let (x, y) = viewer.viewports[panel_index].center();
        viewer
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
            .expect("choose the next empty series panel");
        click_series(&mut viewer, series_index);
    }

    assert_eq!(viewer.viewports.len(), 4);
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.compare_panels.len(), 3);
    let displayed_uids = std::iter::once(
        viewer
            .app
            .loaded
            .as_ref()
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref()),
    )
    .chain(viewer.compare_panels.iter().map(|panel| {
        panel
            .app
            .loaded
            .as_ref()
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref())
    }))
    .collect::<Vec<_>>();
    assert_eq!(
        displayed_uids,
        [
            Some(fixtures::SERIES_UID),
            Some(SECOND_SERIES_UID),
            Some(THIRD_SERIES_UID),
            Some(FOURTH_SERIES_UID),
        ]
    );
    for index in 0..4 {
        let (x, y) = viewer.viewports[index].center();
        assert_ne!(
            viewer.framebuffer.get_pixel(x, y),
            metis_platform::Color::BLACK,
            "panel {index} must display its assigned series"
        );
    }
}

#[test]
fn dragging_a_series_card_assigns_only_the_drop_target_panel() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 2).expect("four-panel grid is within the picker bounds"),
        ))
        .expect("select four-panel layout");
    viewer.refresh_frame().expect("render four panel targets");

    drag_series_to_panel(&mut viewer, 2, 3);

    assert_eq!(viewer.active_panel, 3);
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.compare_panels[0].series_index, None);
    assert_eq!(viewer.compare_panels[1].series_index, None);
    assert_eq!(viewer.compare_panels[2].series_index, Some(2));
    assert_eq!(
        viewer.compare_panels[2]
            .app
            .loaded
            .as_ref()
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref()),
        Some(THIRD_SERIES_UID)
    );
}

#[test]
fn initial_comparison_loads_distinct_requested_series_into_both_panels() {
    let root = replacement_study();
    let tree = scan_folder_for_series(root.path()).expect("scan comparison study");
    let browser =
        SeriesBrowser::from_tree(&tree, Some(fixtures::SERIES_UID)).expect("select primary series");
    let primary = load_volume_from_series_info(
        &browser
            .choice(browser.active_index())
            .expect("primary series")
            .acquisition,
    )
    .expect("load primary series");
    let mut app = crate::app::SnapApp::default();
    app.load_volume(primary, "primary".to_owned());
    let viewer = crate::presentation::native_session::NativeViewerSession::new_with_browser(
        app,
        std::sync::Arc::new(
            crate::presentation::native_session::NativeViewerObservation::default(),
        ),
        false,
        crate::launch::NativePresentationSelection::Fixed(
            crate::launch::NativePresentationMode::Orthogonal,
        ),
        false,
        Some(browser),
        Some(SECOND_SERIES_UID),
    )
    .expect("initialize two-series comparison");

    assert!(viewer.workspace_layout.is_grid());
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(
        viewer
            .compare_panels
            .first()
            .and_then(|panel| panel.series_index),
        Some(1)
    );
    assert_eq!(
        viewer
            .app
            .loaded
            .as_ref()
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref()),
        Some(fixtures::SERIES_UID)
    );
    assert_eq!(
        viewer
            .compare_panels
            .first()
            .and_then(|panel| panel.app.loaded.as_ref())
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref()),
        Some(SECOND_SERIES_UID)
    );
}
