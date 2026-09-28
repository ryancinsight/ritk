//! Series-grid geometry and bounds tests.

use super::*;
use crate::presentation::native_session::frame::empty_axial_view;

fn grid(columns: u32, rows: u32) -> PanelGrid {
    PanelGrid::new(columns, rows).expect("grid dimensions are within the picker bounds")
}

#[test]
fn twenty_panel_layout_places_every_panel_inside_the_surface() {
    let placeholder = empty_axial_view().expect("empty axial frame");
    let panel = GridPanel {
        view: None,
        label: "Select a series",
        navigation: (1.0, ViewportOffset::new(0.0, 0.0)),
    };
    let panels = [panel; MAX_GRID_PANELS];

    let grid = surface_frames_grid(
        &panels,
        grid(5, 4),
        19,
        &placeholder,
        [1_280, 800],
        ViewportArea {
            x: 252,
            y: 88,
            width: 1_028,
            height: 686,
        },
    )
    .expect("render the full 5 by 4 series grid");

    assert_eq!(grid.viewports.len(), 20);
    assert!(grid.viewports[0].contains(300.0, 140.0));
    assert!(grid.viewports[19].contains(1_150.0, 700.0));
    assert!(!grid.viewports[0].contains(1_150.0, 700.0));
    assert_eq!(grid.framebuffer.get_pixel(1_150, 700), WORKSPACE_BACKGROUND);
}

#[test]
fn vertical_two_panel_layout_stacks_without_overlapping() {
    let placeholder = empty_axial_view().expect("empty axial frame");
    let panels = [
        GridPanel {
            view: None,
            label: "P1",
            navigation: (1.0, ViewportOffset::new(0.0, 0.0)),
        },
        GridPanel {
            view: None,
            label: "P2",
            navigation: (1.0, ViewportOffset::new(0.0, 0.0)),
        },
    ];

    let grid = surface_frames_grid(
        &panels,
        grid(1, 2),
        1,
        &placeholder,
        [400, 500],
        ViewportArea::full(400, 500),
    )
    .expect("render two stacked series panels");

    assert_eq!(grid.viewports.len(), 2);
    assert!(grid.viewports[0].panel_y < grid.viewports[1].panel_y);
    assert!(!grid.viewports[0].contains(200.0, 400.0));
    assert!(grid.viewports[1].contains(200.0, 400.0));
}

#[test]
fn twenty_panel_layout_rejects_a_surface_that_cannot_fit_headers() {
    let placeholder = empty_axial_view().expect("empty axial frame");
    let panel = GridPanel {
        view: None,
        label: "P",
        navigation: (1.0, ViewportOffset::new(0.0, 0.0)),
    };
    let panels = [panel; MAX_GRID_PANELS];

    let result = surface_frames_grid(
        &panels,
        grid(5, 4),
        0,
        &placeholder,
        [50, 50],
        ViewportArea::full(50, 50),
    );
    let error = match result {
        Ok(_) => panic!("reject cells shorter than their headers"),
        Err(error) => error,
    };

    assert_eq!(
        error.to_string(),
        "native surface cannot allocate visible series-grid panels"
    );
}

#[test]
fn panel_grid_accepts_radiant_bounds_and_rejects_invalid_dimensions() {
    for rows in 1..=MAX_GRID_ROWS {
        for columns in 1..=MAX_GRID_COLUMNS {
            let grid = grid(columns, rows);
            assert_eq!(grid.dimensions(), (columns, rows));
            assert_eq!(
                grid.panel_count(),
                usize::try_from(columns * rows).expect("grid count fits in usize")
            );
            assert!(grid.menu_label().is_some());
        }
    }
    assert_eq!(grid(5, 4).panel_count(), MAX_GRID_PANELS);
    assert_eq!(grid(3, 2).menu_label(), Some("3 x 2"));
    assert_eq!(PanelGrid::new(0, 1), None);
    assert_eq!(PanelGrid::new(1, 0), None);
    assert_eq!(PanelGrid::new(6, 1), None);
    assert_eq!(PanelGrid::new(1, 5), None);
}

#[test]
fn panel_grid_selects_the_smallest_wide_layout_containing_a_panel() {
    for panel_index in 0..MAX_GRID_PANELS {
        let grid = PanelGrid::containing_panel(panel_index)
            .expect("every supported panel index has a containing layout");
        assert!(grid.panel_count() > panel_index);
    }
    assert_eq!(
        PanelGrid::containing_panel(3).map(PanelGrid::dimensions),
        Some((4, 1))
    );
    assert_eq!(
        PanelGrid::containing_panel(5).map(PanelGrid::dimensions),
        Some((3, 2))
    );
    assert_eq!(
        PanelGrid::containing_panel(19).map(PanelGrid::dimensions),
        Some((5, 4))
    );
    assert_eq!(PanelGrid::containing_panel(MAX_GRID_PANELS), None);
}
