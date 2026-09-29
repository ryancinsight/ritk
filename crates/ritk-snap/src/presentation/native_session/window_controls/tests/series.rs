use super::*;
use crate::presentation::{PointerButton, PresentationEvent};

#[test]
fn visible_workspace_places_the_series_preview_to_the_left_of_the_image_panels() {
    let app = SnapApp::default();
    let layout = ChromeLayout::new(1_280, 800, None, &app, true, WorkspaceLayout::Orthogonal)
        .expect("chrome layout");
    let pan_action = WindowAction::SelectTool(ToolKind::Pan);
    let (pan_x, pan_y) = layout
        .action_center(pan_action)
        .expect("Pan control is visible in the toolbar");
    assert_eq!(
        layout.viewport_area(),
        ViewportArea {
            x: 276,
            y: 88,
            width: 1_004,
            height: 686
        }
    );
    assert_eq!(
        layout.series_preview_area(),
        metis_platform::Rect::new(0, 88, 276, 686)
    );
    assert_eq!(layout.visible_series(), 4);
    assert!(!layout.owns_pointer(640.0, 400.0));
    assert!(layout.owns_pointer(100.0, 160.0));
    assert!(layout.owns_pointer(640.0, 780.0));
    assert_eq!(
        layout.action_at(20.0, 48.0, None),
        Some(WindowAction::OpenStudy)
    );
    assert_eq!(
        layout.action_at(f64::from(pan_x), f64::from(pan_y), None),
        Some(pan_action)
    );
    assert_eq!(layout.action_at(10.0, 799.0, None), None);
}

#[test]
fn window_menu_shortcuts_match_panel_actions() {
    let app = SnapApp::default();
    let mut browser = None;
    let mut chrome = WindowChrome::new(true);
    let grid = WorkspaceLayout::Panels(PanelGrid::new(2, 1).expect("two-panel grid"));
    let ctrl = crate::presentation::PresentationModifiers::new(true, false, false, false);
    let shift = crate::presentation::PresentationModifiers::new(false, true, false, false);
    let cases = [
        (0x4d, ctrl, WindowAction::ToggleActivePanel),
        (0x73, ctrl, WindowAction::CloseActivePanel),
        (0x73, shift, WindowAction::CloseAllPanels),
        (
            0x09,
            crate::presentation::PresentationModifiers::NONE,
            WindowAction::ActivateNextPanel,
        ),
        (0x09, shift, WindowAction::ActivatePreviousPanel),
        (
            0x73,
            crate::presentation::PresentationModifiers::NONE,
            WindowAction::OpenSeriesPicker,
        ),
    ];
    for (virtual_key, modifiers, expected) in cases {
        let event = PresentationEvent::KeyDown {
            virtual_key,
            repeated: false,
            modifiers,
        };
        assert_eq!(
            chrome
                .handle_event(&event, 1_280, 800, &app, &mut browser, grid, false, &[])
                .expect("translate a documented window shortcut"),
            WindowChromeEvent {
                consumed: true,
                repaint: true,
                action: Some(expected),
            }
        );
    }
}

#[test]
fn series_preview_hides_before_it_collapses_the_viewport() {
    let app = SnapApp::default();
    let hidden = ChromeLayout::new(640, 479, None, &app, true, WorkspaceLayout::Orthogonal)
        .expect("hide the rail below its width threshold");
    assert_eq!(hidden.series_preview_area().width, 0);
    assert_eq!(hidden.viewport_area().height, 365);

    let visible = ChromeLayout::new(640, 480, None, &app, true, WorkspaceLayout::Orthogonal)
        .expect("keep the rail visible when the viewer minimum size fits");
    assert_eq!(
        visible.series_preview_area(),
        metis_platform::Rect::new(0, 88, 276, 366)
    );
    assert_eq!(
        visible.viewport_area(),
        ViewportArea {
            x: 276,
            y: 88,
            width: 364,
            height: 366,
        }
    );
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
                false,
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
fn grouped_series_cards_route_selection_and_consume_viewer_input() {
    let mut chrome = WindowChrome::new(true);
    let app = SnapApp::default();
    let (series_browser, _root) = browser_with_series(2);
    let mut browser = Some(series_browser);
    let layout = ChromeLayout::new(1_280, 800, None, &app, true, WorkspaceLayout::Orthogonal)
        .expect("right series preview");
    let series_browser = browser.as_ref().expect("retained series catalog");
    let first_card = super::super::series::navigator::card_center(
        series_browser,
        layout.series_preview_area(),
        0,
    )
    .expect("first series card geometry");
    let second_card = super::super::series::navigator::card_center(
        series_browser,
        layout.series_preview_area(),
        1,
    )
    .expect("second series card geometry");
    assert_eq!(
        layout.series_index_at(
            browser.as_ref(),
            f64::from(first_card.0),
            f64::from(first_card.1)
        ),
        Some(0),
    );
    assert_eq!(
        layout.series_index_at(
            browser.as_ref(),
            f64::from(second_card.0),
            f64::from(second_card.1)
        ),
        Some(1),
    );
    assert_eq!(layout.series_index_at(browser.as_ref(), 20.0, 100.0), None);
    let event = PresentationEvent::PointerDown {
        x: f64::from(second_card.0),
        y: f64::from(second_card.1),
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
                false,
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
                    x: f64::from(second_card.0),
                    y: f64::from(second_card.1),
                    button: PointerButton::Left,
                },
                1_280,
                800,
                &app,
                &mut browser,
                WorkspaceLayout::Orthogonal,
                false,
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
fn vertical_scrollbar_pages_through_the_discovered_series() {
    let mut chrome = WindowChrome::new(true);
    let app = SnapApp::default();
    let (series_browser, _root) = browser_with_series(20);
    let mut browser = Some(series_browser);

    assert_eq!(
        chrome
            .handle_event(
                &PresentationEvent::PointerDown {
                    x: 270.0,
                    y: 750.0,
                    button: PointerButton::Left,
                },
                1_280,
                800,
                &app,
                &mut browser,
                WorkspaceLayout::Orthogonal,
                false,
                &[],
            )
            .expect("page down through the series list"),
        WindowChromeEvent::consumed(true)
    );
    assert_eq!(browser.as_ref().map(SeriesBrowser::first_visible), Some(4));
}

#[test]
fn preview_wheel_scrolls_through_the_discovered_series() {
    let mut chrome = WindowChrome::new(true);
    let app = SnapApp::default();
    let (series_browser, _root) = browser_with_series(10);
    let mut browser = Some(series_browser);
    let event = PresentationEvent::PointerWheel {
        x: 100.0,
        y: 160.0,
        delta_x: 0.0,
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
                false,
                &[],
            )
            .expect("scroll the series preview"),
        WindowChromeEvent::consumed(true)
    );
    assert_eq!(browser.as_ref().expect("series preview").first_visible(), 3);
}
