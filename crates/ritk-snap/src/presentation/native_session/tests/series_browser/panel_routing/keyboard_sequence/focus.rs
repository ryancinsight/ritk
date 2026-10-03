use super::*;

#[test]
fn queued_series_browse_uses_pointer_focused_panel_after_tab() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("side-by-side grid is valid"),
        ))
        .expect("select side-by-side layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series into panel two");
    viewer.refresh_frame().expect("render both series");
    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x09,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("focus the primary panel");
    assert_eq!(viewer.active_panel, 0);
    let (x, y) = viewer.viewports[1].center();

    viewer
        .handle_events(&[
            WindowEvent::PointerWheel {
                x,
                y,
                delta_x: 0,
                delta_y: -120,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x09,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x27,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x27,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("follow the pointer-focused panel through Tab and series browsing");

    assert_eq!(viewer.active_panel, 0);
    assert_eq!(viewer.primary_series_index, Some(2));
    assert_eq!(viewer.compare_panels[0].series_index, Some(1));
    assert_eq!(loaded_series_uid(&viewer.app), Some(THIRD_SERIES_UID));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(SECOND_SERIES_UID)
    );
}

#[test]
fn queued_image_navigation_uses_pointer_moved_panel_after_tab() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("side-by-side grid is valid"),
        ))
        .expect("select side-by-side layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series into panel two");
    viewer.refresh_frame().expect("render both series");
    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x09,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("focus the primary panel");
    let primary_slice = viewer.app.viewer_state.slice_index;
    let secondary_slice = viewer.compare_panels[0].app.viewer_state.slice_index;
    let (x, y) = viewer.viewports[1].center();

    viewer
        .handle_events(&[
            WindowEvent::PointerMove { x, y },
            WindowEvent::KeyDown {
                virtual_key: 0x09,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x28,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("follow the pointer-moved panel through Tab and image navigation");

    assert_eq!(viewer.active_panel, 0);
    assert_ne!(viewer.app.viewer_state.slice_index, primary_slice);
    assert_eq!(
        viewer.compare_panels[0].app.viewer_state.slice_index,
        secondary_slice
    );
}
