//! Native RadiAnt-clone workspace geometry and input tests.

use super::*;
use crate::dicom::loader::{scan_folder_for_series, tests::fixtures};
use crate::presentation::native_session::layout::{PanelGrid, WorkspaceLayout};

fn browser_with_series(count: usize) -> (SeriesBrowser, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("study root");
    for index in 0..count {
        fixtures::write_study(root.path(), "MR", &format!("2.25.20260905{index:04}"))
            .expect("write series");
    }
    let tree = scan_folder_for_series(root.path()).expect("scan study");
    let browser = SeriesBrowser::from_tree(&tree, None).expect("series browser");
    (browser, root)
}

#[test]
fn visible_workspace_reserves_the_bottom_series_preview_bar() {
    let app = SnapApp::default();
    let layout = ChromeLayout::new(1_280, 800, None, &app, true, WorkspaceLayout::Orthogonal)
        .expect("chrome layout");
    assert_eq!(
        layout.viewport_area(),
        ViewportArea {
            x: 0,
            y: 88,
            width: 1_280,
            height: 554
        }
    );
    assert_eq!(
        layout.series_preview_area(),
        metis_platform::Rect::new(0, 642, 1_280, 132)
    );
    assert_eq!(layout.visible_series(), 6);
    assert!(!layout.owns_pointer(640.0, 400.0));
    assert!(layout.owns_pointer(640.0, 700.0));
    assert!(layout.owns_pointer(640.0, 780.0));
    assert_eq!(
        layout.action_at(20.0, 48.0, None),
        Some(WindowAction::OpenStudy)
    );
    assert_eq!(
        layout.action_at(200.0, 48.0, None),
        Some(WindowAction::SelectTool(ToolKind::Pan))
    );
    assert_eq!(layout.action_at(10.0, 799.0, None), None);
}

#[test]
fn hidden_capture_keeps_the_pane_only_viewport() {
    let chrome = WindowChrome::new(false);
    assert_eq!(
        chrome.viewport_area(1_280, 800).expect("hidden viewport"),
        ViewportArea::full(1_280, 800)
    );
}

#[test]
fn focus_loss_closes_an_open_menu_and_requests_a_repaint() {
    let mut chrome = WindowChrome::new(true);
    chrome.open_menu = Some(Menu::View);
    let app = SnapApp::default();
    let mut browser = None;

    assert_eq!(
        chrome
            .handle_event(
                &PresentationEvent::FocusLost,
                1_280,
                800,
                &app,
                &mut browser,
                WorkspaceLayout::Orthogonal,
                &[],
            )
            .expect("close the menu on focus loss"),
        WindowChromeEvent {
            consumed: false,
            repaint: true,
            action: None,
        }
    );
    assert_eq!(chrome.open_menu, None);
}

#[test]
fn view_menu_routes_panel_and_crosshair_commands() {
    let app = SnapApp::default();
    let layout = ChromeLayout::new(
        1_280,
        800,
        Some(Menu::View),
        &app,
        true,
        WorkspaceLayout::Orthogonal,
    )
    .expect("View menu layout");
    assert_eq!(
        layout.action_at(80.0, 44.0, None),
        Some(WindowAction::ToggleSeriesPreview)
    );
    assert_eq!(
        layout.action_at(80.0, 72.0, None),
        Some(WindowAction::ToggleCrosshair)
    );
}

#[test]
fn horizontal_series_cards_route_selection_and_consume_viewer_input() {
    let mut chrome = WindowChrome::new(true);
    let app = SnapApp::default();
    let (series_browser, _root) = browser_with_series(2);
    let mut browser = Some(series_browser);
    let layout = ChromeLayout::new(1_280, 800, None, &app, true, WorkspaceLayout::Orthogonal)
        .expect("horizontal series preview");
    assert_eq!(
        layout.series_index_at(browser.as_ref(), 32.0, 700.0),
        Some(0)
    );
    assert_eq!(
        layout.series_index_at(browser.as_ref(), 232.0, 700.0),
        Some(1)
    );
    assert_eq!(layout.series_index_at(browser.as_ref(), 195.0, 700.0), None);
    let event = PresentationEvent::PointerDown {
        x: 232.0,
        y: 700.0,
        button: PointerButton::Left,
    };

    assert_eq!(
        chrome
            .handle_event(
                &event,
                1_280,
                800,
                &app,
                &mut browser,
                WorkspaceLayout::Orthogonal,
                &[],
            )
            .expect("begin a series drag"),
        WindowChromeEvent {
            consumed: true,
            repaint: false,
            action: None,
        }
    );
    assert_eq!(
        chrome
            .handle_event(
                &PresentationEvent::PointerUp {
                    x: 232.0,
                    y: 700.0,
                    button: PointerButton::Left,
                },
                1_280,
                800,
                &app,
                &mut browser,
                WorkspaceLayout::Orthogonal,
                &[],
            )
            .expect("select a series row"),
        WindowChromeEvent {
            consumed: true,
            repaint: true,
            action: Some(WindowAction::SelectSeries(1)),
        }
    );
}

#[test]
fn preview_wheel_scrolls_the_discovered_series_list() {
    let mut chrome = WindowChrome::new(true);
    let app = SnapApp::default();
    let (series_browser, _root) = browser_with_series(10);
    let mut browser = Some(series_browser);
    let event = PresentationEvent::PointerWheel {
        x: 40.0,
        y: 700.0,
        delta_x: 120.0,
        delta_y: -120.0,
        modifiers: crate::presentation::PresentationModifiers::NONE,
    };

    assert_eq!(
        chrome
            .handle_event(
                &event,
                1_280,
                800,
                &app,
                &mut browser,
                WorkspaceLayout::Orthogonal,
                &[],
            )
            .expect("scroll the series preview"),
        WindowChromeEvent::consumed(true)
    );
    assert_eq!(browser.as_ref().expect("series preview").first_visible(), 3);
}

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
fn window_menu_opens_the_radiant_grid_picker() {
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

    assert_eq!(
        layout.action_at(200.0, 44.0, None),
        Some(WindowAction::SetLayout(WorkspaceLayout::Orthogonal))
    );
    assert_eq!(
        layout.action_at(200.0, 72.0, None),
        Some(WindowAction::OpenMenu(Menu::GridPicker))
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
            let x = 578.0 + f64::from(columns.saturating_sub(1)) * 54.0;
            let y = 151.0 + f64::from(rows.saturating_sub(1)) * 38.0;
            let grid = PanelGrid::new(columns, rows).expect("picker dimensions are valid");
            assert_eq!(
                layout.action_at(x, y, None),
                Some(WindowAction::SetLayout(WorkspaceLayout::Panels(grid))),
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
fn tools_popup_takes_pointer_priority_over_series_cards() {
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
            &[],
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
                &[],
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
            &[],
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
            &[],
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
                &[],
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
            y: 88,
            width: 1_280,
            height: 686,
        }
    );
}
