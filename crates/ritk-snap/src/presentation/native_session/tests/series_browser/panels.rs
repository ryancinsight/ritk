use super::*;

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
fn one_input_batch_routes_wheels_to_each_comparison_panel() {
    let (mut viewer, _initial_root) = session();
    let study = replacement_study();
    viewer
        .open_study_path(study.path())
        .expect("open two-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("side-by-side grid is valid"),
        ))
        .expect("select side-by-side layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series into panel two");
    viewer.refresh_frame().expect("render both series");

    let primary_before = viewer.app.viewer_state.slice_index;
    let secondary_before = viewer.compare_panels[0].app.viewer_state.slice_index;
    let (left_x, left_y) = viewer.viewports[0].center();
    let (right_x, right_y) = viewer.viewports[1].center();
    viewer
        .handle_events(&[
            WindowEvent::PointerWheel {
                x: left_x,
                y: left_y,
                delta_x: 0,
                delta_y: -120,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::PointerWheel {
                x: right_x,
                y: right_y,
                delta_x: 0,
                delta_y: -120,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("route both panel wheels in one host batch");

    assert_ne!(viewer.app.viewer_state.slice_index, primary_before);
    assert_ne!(
        viewer.compare_panels[0].app.viewer_state.slice_index,
        secondary_before
    );
    assert_eq!(viewer.active_panel, 1);
}

#[test]
fn rejected_comparison_batch_does_not_apply_an_earlier_panel_event() {
    let (mut viewer, _initial_root) = session();
    let study = replacement_study();
    viewer
        .open_study_path(study.path())
        .expect("open two-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("side-by-side grid is valid"),
        ))
        .expect("select side-by-side layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series into panel two");
    viewer.refresh_frame().expect("render both series");

    let primary_before = viewer.app.viewer_state.slice_index;
    let secondary_before = viewer.compare_panels[0].app.viewer_state.slice_index;
    let (left_x, left_y) = viewer.viewports[0].center();
    let (right_x, right_y) = viewer.viewports[1].center();
    let error = viewer
        .handle_events(&[
            WindowEvent::PointerWheel {
                x: left_x,
                y: left_y,
                delta_x: 0,
                delta_y: -120,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::PointerUp {
                x: right_x,
                y: right_y,
                button: MouseButton::Left,
            },
        ])
        .expect_err("reject release without a press in the second panel");

    assert!(error.to_string().contains("without a press"));
    assert_eq!(viewer.app.viewer_state.slice_index, primary_before);
    assert_eq!(
        viewer.compare_panels[0].app.viewer_state.slice_index,
        secondary_before
    );
}

#[test]
fn rejected_native_batch_does_not_apply_pane_events_before_toolbar_actions() {
    let (mut viewer, _initial_root) = session();
    let study = replacement_study();
    viewer
        .open_study_path(study.path())
        .expect("open two-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("side-by-side grid is valid"),
        ))
        .expect("select side-by-side layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series into panel two");
    viewer.active_panel = 0;
    viewer.refresh_frame().expect("render both series");
    viewer.app.active_tool = crate::tools::kind::ToolKind::WindowLevel;

    let primary_slice = viewer.app.viewer_state.slice_index;
    let primary_tool = viewer.app.active_tool;
    let (left_x, left_y) = viewer.viewports[0].center();
    let (right_x, right_y) = viewer.viewports[1].center();
    let error = viewer
        .handle_events(&[
            WindowEvent::PointerWheel {
                x: left_x,
                y: left_y,
                delta_x: 0,
                delta_y: -120,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::PointerDown {
                x: 200,
                y: 48,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: 200,
                y: 48,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: right_x,
                y: right_y,
                button: MouseButton::Left,
            },
        ])
        .expect_err("reject the unpressed release after all earlier inputs");

    assert!(error.to_string().contains("without a press"));
    assert_eq!(viewer.app.viewer_state.slice_index, primary_slice);
    assert_eq!(viewer.app.active_tool, primary_tool);
}

#[test]
fn pane_input_after_layout_selection_uses_the_presented_layout_snapshot() {
    let (mut viewer, _initial_root) = session();
    viewer
        .refresh_frame()
        .expect("render the orthogonal workspace");
    let (sagittal_x, sagittal_y) = viewer.viewports[2].center();

    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: 690,
                y: 65,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: 690,
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
            WindowEvent::PointerDown {
                x: sagittal_x,
                y: sagittal_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: sagittal_x,
                y: sagittal_y,
                button: MouseButton::Left,
            },
        ])
        .expect("reduce pane events against the layout visible at batch start");

    assert!(viewer.workspace_layout.is_grid());
    assert_eq!(viewer.viewports.len(), 2);
    assert_eq!(viewer.app.axis, 2);
}

#[test]
fn layout_change_cancels_a_captured_drag_before_remapping_panes() {
    let (mut viewer, _initial_root) = session();
    viewer
        .refresh_frame()
        .expect("render the orthogonal workspace");
    let (sagittal_x, sagittal_y) = viewer.viewports[2].center();
    viewer.app.active_tool = crate::tools::kind::ToolKind::Pan;

    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: 690,
                y: 65,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: 690,
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
            WindowEvent::PointerDown {
                x: sagittal_x,
                y: sagittal_y,
                button: MouseButton::Left,
            },
        ])
        .expect("apply layout selection against the presented orthogonal frame");

    assert!(viewer.workspace_layout.is_grid());
    assert_eq!(viewer.viewports.len(), 2);
    assert_eq!(viewer.active_view, None);
    assert_eq!(
        viewer.suppress_cancelled_pointer_release,
        Some(crate::presentation::PointerButton::Left)
    );
    assert!(viewer.app.tool_state.is_idle());

    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: 690,
                y: 65,
                button: MouseButton::Right,
            },
            WindowEvent::PointerUp {
                x: 690,
                y: 65,
                button: MouseButton::Right,
            },
        ])
        .expect("a different button does not release the canceled left gesture");
    assert_eq!(
        viewer.suppress_cancelled_pointer_release,
        Some(crate::presentation::PointerButton::Left)
    );

    viewer
        .handle_events(&[WindowEvent::PointerMove {
            x: sagittal_x + 16,
            y: sagittal_y + 12,
        }])
        .expect("ignore the canceled drag while the new layout is active");
    viewer
        .handle_events(&[WindowEvent::PointerUp {
            x: sagittal_x + 16,
            y: sagittal_y + 12,
            button: MouseButton::Left,
        }])
        .expect("consume the release for the canceled drag");

    assert_eq!(viewer.suppress_cancelled_pointer_release, None);
    assert!(viewer.app.tool_state.is_idle());
}
