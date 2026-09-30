use super::*;
use crate::presentation::{PointerButton, PresentationEvent};

#[test]
fn file_menu_routes_study_open_and_exit_commands() {
    let app = SnapApp::default();
    let layout = ChromeLayout::new(
        1_280,
        800,
        Some(Menu::File),
        &app,
        true,
        WorkspaceLayout::Orthogonal,
    )
    .expect("File menu layout");
    assert_eq!(
        layout.action_at(20.0, 44.0, None),
        Some(WindowAction::OpenStudy)
    );
    assert_eq!(layout.action_at(20.0, 72.0, None), Some(WindowAction::Exit));
}

#[test]
fn toolbar_exposes_the_multiple_series_picker() {
    let app = SnapApp::default();
    let layout = ChromeLayout::new(1_280, 800, None, &app, true, WorkspaceLayout::Orthogonal)
        .expect("native toolbar layout");
    let action = WindowAction::OpenSeriesPicker;
    let (x, y) = layout
        .action_center(action)
        .expect("multiple-series picker is visible in the toolbar");

    assert_eq!(
        layout.action_at(f64::from(x), f64::from(y), None),
        Some(action)
    );
}

#[test]
fn window_menu_opens_the_panel_grid_picker() {
    let app = SnapApp::default();
    let layout = ChromeLayout::new(
        1_280,
        800,
        Some(Menu::Window),
        &app,
        true,
        WorkspaceLayout::Orthogonal,
    )
    .expect("Window menu layout");
    let layout_action = WindowAction::SetLayout(WorkspaceLayout::Orthogonal);
    let (layout_x, layout_y) = layout
        .action_center(layout_action)
        .expect("orthogonal layout menu item is visible");
    let picker_action = WindowAction::OpenMenu(Menu::GridPicker);
    let (picker_x, picker_y) = layout
        .action_center(picker_action)
        .expect("panel grid picker menu item is visible");

    assert_eq!(
        layout.action_at(f64::from(layout_x), f64::from(layout_y), None),
        Some(layout_action)
    );
    assert_eq!(
        layout.action_at(f64::from(picker_x), f64::from(picker_y), None),
        Some(picker_action)
    );
}

#[test]
fn grid_picker_selects_every_supported_column_and_row_extent() {
    let app = SnapApp::default();
    let layout = ChromeLayout::new(
        1_280,
        800,
        Some(Menu::GridPicker),
        &app,
        true,
        WorkspaceLayout::Orthogonal,
    )
    .expect("grid picker layout");

    for rows in 1..=4_u32 {
        for columns in 1..=5_u32 {
            let grid = PanelGrid::new(columns, rows).expect("picker dimensions are valid");
            let action = WindowAction::SetLayout(WorkspaceLayout::Panels(grid));
            let (x, y) = layout
                .action_center(action)
                .expect("each supported grid has a visible picker cell");
            assert_eq!(
                layout.action_at(f64::from(x), f64::from(y), None),
                Some(action),
                "picker cell {columns} x {rows} selects its matching grid",
            );
        }
    }
    assert!(!layout.owns_pointer(500.0, 500.0));
}

#[test]
fn tools_menu_hit_tests_each_ritk_tool() {
    let app = SnapApp::default();
    let layout = ChromeLayout::new(
        1_280,
        800,
        Some(Menu::Tools),
        &app,
        true,
        WorkspaceLayout::Orthogonal,
    )
    .expect("Tools menu layout");
    for (index, tool) in ToolKind::all().iter().copied().enumerate() {
        let y = 44.0 + f64::from(u32::try_from(index).expect("bounded menu index")) * 28.0;
        assert_eq!(
            layout.action_at(130.0, y, None),
            Some(WindowAction::SelectTool(tool))
        );
    }
}

#[test]
fn panel_layout_hides_crosshair_controls_that_are_not_rendered() {
    let app = SnapApp::default();
    let workspace = WorkspaceLayout::Panels(PanelGrid::new(2, 1).expect("two-panel layout"));
    let toolbar = ChromeLayout::new(1_280, 800, None, &app, true, workspace)
        .expect("comparison toolbar layout");
    assert_eq!(
        toolbar.action_center(WindowAction::SelectTool(ToolKind::Crosshair)),
        None
    );

    for (menu, action) in [
        (Menu::View, WindowAction::ToggleCrosshair),
        (Menu::Tools, WindowAction::SelectTool(ToolKind::Crosshair)),
    ] {
        let layout = ChromeLayout::new(1_280, 800, Some(menu), &app, true, workspace)
            .expect("comparison menu layout");
        assert_eq!(layout.action_center(action), None);
    }

    let view_menu = ChromeLayout::new(1_280, 800, Some(Menu::View), &app, true, workspace)
        .expect("comparison View menu layout");
    for action in [WindowAction::ToggleCine, WindowAction::ResetView] {
        let (x, y) = view_menu
            .action_center(action)
            .expect("View menu action has a visible hit target");
        assert_eq!(
            view_menu.action_at(f64::from(x), f64::from(y), None),
            Some(action)
        );
    }
}

#[test]
fn tools_popup_consumes_pointer_without_changing_series_selection() {
    let mut chrome = WindowChrome::new(true);
    chrome.open_menu = Some(Menu::Tools);
    let app = SnapApp::default();
    let (series_browser, _root) = browser_with_series(2);
    let mut browser = Some(series_browser);
    assert!(browser.as_mut().expect("series browser").set_active(1));

    let result = chrome
        .handle_event(
            &PresentationEvent::PointerDown {
                x: 160.0,
                y: 156.0,
                button: PointerButton::Left,
            },
            1_280,
            800,
            &app,
            &mut browser,
            WorkspaceLayout::Orthogonal,
            false,
            &[],
            &[],
            0,
        )
        .expect("select the obscured Tools menu item");

    assert_eq!(
        result.action,
        Some(WindowAction::SelectTool(ToolKind::MeasureAngle))
    );
    assert_eq!(browser.as_ref().expect("series browser").active_index(), 1);
    assert_eq!(chrome.open_menu, None);
}

#[test]
fn wheel_over_tools_popup_does_not_scroll_the_series_list() {
    let mut chrome = WindowChrome::new(true);
    chrome.open_menu = Some(Menu::Tools);
    let app = SnapApp::default();
    let (series_browser, _root) = browser_with_series(10);
    let mut browser = Some(series_browser);
    let before = browser.as_ref().expect("series browser").first_visible();

    assert_eq!(
        chrome
            .handle_event(
                &PresentationEvent::PointerWheel {
                    x: 160.0,
                    y: 156.0,
                    delta_x: 0.0,
                    delta_y: -120.0,
                    modifiers: crate::presentation::PresentationModifiers::NONE,
                },
                1_280,
                800,
                &app,
                &mut browser,
                WorkspaceLayout::Orthogonal,
                false,
                &[],
                &[],
                0,
            )
            .expect("keep wheel input in the menu layer"),
        WindowChromeEvent::consumed(false)
    );
    assert_eq!(
        browser.as_ref().expect("series browser").first_visible(),
        before
    );
}

#[test]
fn narrow_windows_keep_every_control_inside_the_surface() {
    let app = SnapApp::default();
    let layout = ChromeLayout::new(
        72,
        54,
        Some(Menu::Tools),
        &app,
        true,
        WorkspaceLayout::Orthogonal,
    )
    .expect("constrained chrome layout");
    assert_eq!(layout.viewport_area().width, 72);
    assert_eq!(layout.viewport_area().height, 0);
    assert!(layout.controls_fit_within(72, 54));
}

#[test]
fn native_menu_event_toggles_the_series_preview() {
    let mut chrome = WindowChrome::new(true);
    let app = SnapApp::default();
    let mut browser = None;
    let menu_down = PresentationEvent::PointerDown {
        x: 80.0,
        y: 12.0,
        button: PointerButton::Left,
    };
    let menu_up = PresentationEvent::PointerUp {
        x: 80.0,
        y: 12.0,
        button: PointerButton::Left,
    };
    chrome
        .handle_event(
            &menu_down,
            1_280,
            800,
            &app,
            &mut browser,
            WorkspaceLayout::Orthogonal,
            false,
            &[],
            &[],
            0,
        )
        .expect("open View menu");
    chrome
        .handle_event(
            &menu_up,
            1_280,
            800,
            &app,
            &mut browser,
            WorkspaceLayout::Orthogonal,
            false,
            &[],
            &[],
            0,
        )
        .expect("release View menu");
    let toggle = PresentationEvent::PointerDown {
        x: 80.0,
        y: 44.0,
        button: PointerButton::Left,
    };

    assert_eq!(
        chrome
            .handle_event(
                &toggle,
                1_280,
                800,
                &app,
                &mut browser,
                WorkspaceLayout::Orthogonal,
                false,
                &[],
                &[],
                0,
            )
            .expect("toggle series preview"),
        WindowChromeEvent {
            consumed: true,
            repaint: true,
            action: Some(WindowAction::ToggleSeriesPreview),
        }
    );
    chrome.toggle_series_preview();
    assert_eq!(
        chrome
            .viewport_area(1_280, 800)
            .expect("viewport without series preview"),
        ViewportArea {
            x: 0,
            y: 78,
            width: 1_280,
            height: 696,
        }
    );
}
