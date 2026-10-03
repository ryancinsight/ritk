//! Series-grid geometry and bounds tests.

use super::*;
use crate::presentation::native_session::frame::empty_axial_view;
use crate::presentation::native_session::window_controls::WindowChrome;
use crate::render::WindowLevel;

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
        maximized: false,
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
            maximized: false,
        },
        GridPanel {
            view: None,
            label: "P2",
            navigation: (1.0, ViewportOffset::new(0.0, 0.0)),
            maximized: false,
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
fn compact_four_row_layout_preserves_populated_panel_geometry() {
    let view = empty_axial_view().expect("empty axial frame");
    let panel = GridPanel {
        view: Some(&view),
        label: "P",
        navigation: (1.0, ViewportOffset::new(0.0, 0.0)),
        maximized: false,
    };
    let panels = [panel; 4];
    let viewport_area = WindowChrome::new(true)
        .viewport_area(640, 220)
        .expect("derive the image area from native window chrome");

    let rendered = surface_frames_grid(&panels, grid(1, 4), 3, &view, [640, 220], viewport_area)
        .expect("render four populated rows in the compact native surface");

    let total_gaps = (MAX_GRID_ROWS - 1) * VIEW_GAP_PIXELS;
    let row_height = (viewport_area.height - total_gaps) / MAX_GRID_ROWS;
    assert_eq!(rendered.viewports.len(), panels.len());
    for (row, viewport) in rendered.viewports.iter().enumerate() {
        let row = u32::try_from(row).expect("four rows fit in u32");
        let expected_y = viewport_area.y + row * (row_height + VIEW_GAP_PIXELS);
        assert_eq!(viewport.panel_y, expected_y);
        assert_eq!(viewport.panel_width, viewport_area.width);
        assert_eq!(viewport.panel_height, row_height);
        assert_eq!(viewport.axis(), view.axis);
        assert!(viewport.contains(
            f64::from(viewport_area.width / 2),
            f64::from(expected_y + row_height / 2)
        ));
    }
    let final_viewport = rendered.viewports[3];
    assert_eq!(
        rendered.framebuffer.get_pixel(0, viewport_area.y),
        INACTIVE_PANEL
    );
    assert_eq!(
        rendered.framebuffer.get_pixel(0, final_viewport.panel_y),
        ACTIVE_PANEL
    );
    assert_eq!(
        final_viewport.panel_y + final_viewport.panel_height,
        viewport_area.y + viewport_area.height
    );
}

#[test]
fn twenty_panel_layout_rejects_a_surface_without_nonzero_cells() {
    let placeholder = empty_axial_view().expect("empty axial frame");
    let panel = GridPanel {
        view: None,
        label: "P",
        navigation: (1.0, ViewportOffset::new(0.0, 0.0)),
        maximized: false,
    };
    let panels = [panel; MAX_GRID_PANELS];

    let result = surface_frames_grid(
        &panels,
        grid(5, 4),
        0,
        &placeholder,
        [19, 50],
        ViewportArea::full(19, 50),
    );
    let error = match result {
        Ok(_) => panic!("reject zero-width cells"),
        Err(error) => error,
    };

    assert_eq!(
        error.to_string(),
        "native surface cannot allocate visible series-grid panels"
    );
}

#[test]
fn panel_grid_accepts_supported_bounds_and_rejects_invalid_dimensions() {
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

#[test]
fn grid_panel_image_and_window_values_change_their_rendered_corner_readouts() {
    let mut first_view = empty_axial_view().expect("empty axial frame");
    first_view.slice_index = 17;
    first_view.slice_count = 94;
    first_view.window_level = WindowLevel::new(36.0, 1_204.0);
    let mut next_view = first_view.clone();
    next_view.slice_index = 18;
    next_view.window_level = WindowLevel::new(48.0, 1_640.0);
    let placeholder = empty_axial_view().expect("empty axial frame");
    let render = |view: &RenderedView| {
        surface_frames_grid(
            &[GridPanel {
                view: Some(view),
                label: "P1  |  MR  |  T2",
                navigation: (1.0, ViewportOffset::new(0.0, 0.0)),
                maximized: false,
            }],
            grid(1, 1),
            0,
            &placeholder,
            [320, 240],
            ViewportArea::full(320, 240),
        )
        .expect("render the axial image and its corner values")
        .framebuffer
    };
    let first = render(&first_view);
    let next = render(&next_view);
    let top_readout_changed =
        (8..130).any(|x| (30..50).any(|y| first.get_pixel(x, y) != next.get_pixel(x, y)));
    let bottom_readout_changed =
        (8..150).any(|x| (218..236).any(|y| first.get_pixel(x, y) != next.get_pixel(x, y)));

    assert!(top_readout_changed, "slice navigation updates Image n / N");
    assert!(
        bottom_readout_changed,
        "window/level changes update W and C"
    );
}
