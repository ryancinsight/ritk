use super::*;
use crate::presentation::native_session::layout::{GridPanel, NativeViewport, MAX_GRID_PANELS};
use crate::tools::interaction::ViewportOffset;

fn four_pane_viewports(
    active_panel: usize,
) -> (
    PanelGrid,
    arrayvec::ArrayVec<NativeViewport, MAX_GRID_PANELS>,
) {
    let views = crate::presentation::native_session::frame::empty_orthogonal_views()
        .expect("empty event views");
    let grid = PanelGrid::new(2, 2).expect("four-pane grid");
    let panels: [GridPanel<'_>; 4] = std::array::from_fn(|index| GridPanel {
        view: Some(&views[index.min(2)]),
        label: ["Axial", "Coronal", "Sagittal", "Projection"][index],
        navigation: (1.0, ViewportOffset::new(0.0, 0.0)),
        maximized: false,
    });
    let composition = crate::presentation::native_session::layout::surface_frames_grid(
        &panels,
        grid,
        active_panel,
        &views[0],
        [1_280, 800],
        ViewportArea {
            x: 276,
            y: 78,
            width: 1_004,
            height: 696,
        },
    )
    .expect("four-pane event geometry");
    (grid, composition.viewports)
}

fn horizontal_wheel(center: (i32, i32)) -> PresentationEvent {
    PresentationEvent::PointerWheel {
        x: f64::from(center.0),
        y: f64::from(center.1),
        delta_x: 120.0,
        delta_y: 0.0,
        modifiers: crate::presentation::PresentationModifiers::NONE,
    }
}

#[test]
fn horizontal_wheel_routes_orthogonal_panes_to_their_series_owner() {
    let app = SnapApp::default();
    let (series_browser, _root) = browser_with_series(3);
    let (_, viewports) = four_pane_viewports(0);
    let mut browser = Some(series_browser);

    for (pane, center) in ["axial", "coronal", "sagittal", "projection"]
        .into_iter()
        .zip(viewports.iter().map(|viewport| viewport.center()))
    {
        let mut chrome = WindowChrome::new(true);
        assert_eq!(
            chrome
                .handle_event(
                    &horizontal_wheel(center),
                    1_280,
                    800,
                    &app,
                    &mut browser,
                    WorkspaceLayout::Orthogonal,
                    false,
                    viewports.as_slice(),
                    &[Some(1)],
                    0,
                )
                .unwrap_or_else(|error| panic!("route {pane} series wheel: {error}")),
            WindowChromeEvent {
                consumed: true,
                repaint: true,
                action: Some(WindowAction::BrowseSeries {
                    series_index: 2,
                    panel_index: 0,
                }),
            },
            "{pane} navigates the orthogonal layout's owning series",
        );
    }
}

#[test]
fn horizontal_wheel_preserves_comparison_panel_identity() {
    let app = SnapApp::default();
    let (series_browser, _root) = browser_with_series(3);
    let (grid, viewports) = four_pane_viewports(3);
    let mut chrome = WindowChrome::new(true);
    let mut browser = Some(series_browser);

    assert_eq!(
        chrome
            .handle_event(
                &horizontal_wheel(viewports[3].center()),
                1_280,
                800,
                &app,
                &mut browser,
                WorkspaceLayout::Panels(grid),
                false,
                viewports.as_slice(),
                &[Some(0), Some(0), Some(0), Some(1)],
                3,
            )
            .expect("route comparison series wheel"),
        WindowChromeEvent {
            consumed: true,
            repaint: true,
            action: Some(WindowAction::BrowseSeries {
                series_index: 2,
                panel_index: 3,
            }),
        },
    );
}
