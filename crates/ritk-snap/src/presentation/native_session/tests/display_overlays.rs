//! Native histogram window panel and per-pane colorbar overlays.

use super::super::layout::overlay_display::append_colorbar_overlay;
use super::super::layout::{application_overlay, ClinicalOverlay, OverlayCommands};
use super::support::session;
use iris::color::{ColorMap, Normalized};
use metis_platform::rasterizer::CornerRadius;
use metis_ui_lang::DisplayCommand;

#[test]
fn cached_histogram_counts_every_finite_voxel() {
    let (session, _root) = session();
    let histogram = session
        .app
        .cached_histogram
        .as_ref()
        .expect("histogram cached on study load");
    assert_eq!(histogram.bins, 256);
    let volume = session.app.loaded.as_ref().expect("study loaded");
    let finite = volume.data.iter().filter(|v| v.is_finite()).count();
    assert_eq!(
        histogram.counts.iter().sum::<u64>(),
        u64::try_from(finite).expect("voxel count fits"),
        "histogram bins must partition the finite voxels"
    );
}

#[test]
fn colorbar_endpoints_show_exact_colormap_colors() {
    let (mut session, _root) = session();
    session
        .refresh_frame()
        .expect("frame with display overlays");
    let viewport = session.viewports[0];
    let panel = viewport.panel_rect().expect("pane rect");
    let strip_x = panel.x + panel.width
        - super::super::slice_scrollbar::RIGHT_GUTTER
        - super::super::layout::overlay_display::COLORBAR_WIDTH;
    let image_top = super::super::layout::screen_coordinate(viewport.image.y, "colorbar top")
        .expect("colorbar top fits");
    let image_height =
        super::super::layout::screen_coordinate(viewport.image.height, "colorbar height")
            .expect("colorbar height fits");
    let map = session.app.colormap;
    let expected = |t: u8| {
        let [red, green, blue, _] = map.sample(Normalized::from_u8(t)).to_rgba8();
        u32::from_be_bytes([255, red, green, blue])
    };
    let width = usize::try_from(session.framebuffer.width()).expect("framebuffer width fits");
    let pixel_at = |x: i32, y: i32| {
        session.framebuffer.pixels()
            [usize::try_from(y).expect("row fits") * width + usize::try_from(x).expect("col fits")]
    };
    assert_eq!(
        pixel_at(strip_x + 7, image_top),
        expected(255),
        "colorbar top must sample the colormap maximum"
    );
    assert_eq!(
        pixel_at(strip_x + 7, image_top + image_height - 1),
        expected(0),
        "colorbar bottom must sample the colormap minimum"
    );
}

#[test]
fn histogram_strip_shows_window_mark_and_bars() {
    let (mut session, _root) = session();
    session
        .refresh_frame()
        .expect("frame with display overlays");
    let viewport = session.viewports[0];
    let panel = viewport.panel_rect().expect("pane rect");
    let strip_top =
        panel.y + panel.height - super::super::layout::overlay_display::HISTOGRAM_STRIP_HEIGHT;
    // Strip background is untouched where no bar, band or label reaches.
    let width = usize::try_from(session.framebuffer.width()).expect("framebuffer width fits");
    let pixel_at = |x: i32, y: i32| {
        session.framebuffer.pixels()
            [usize::try_from(y).expect("row fits") * width + usize::try_from(x).expect("col fits")]
    };
    let packed = |channels: [u8; 4]| {
        u32::from_be_bytes([channels[3], channels[0], channels[1], channels[2]])
    };
    assert_eq!(
        pixel_at(panel.x, strip_top + 2),
        packed([20, 20, 20, 255]),
        "histogram strip corner must show the panel background"
    );
    // The opaque centre line marks the window centre over the band.
    let histogram = session
        .app
        .cached_histogram
        .as_ref()
        .expect("histogram cached on study load");
    let (center, width_value) = session.app.browser_window_level_values();
    let canvas_left = panel.x + 8;
    let canvas_right = panel.x + panel.width - 8;
    let center_x = super::super::layout::screen_coordinate(
        f64::from(crate::render::histogram::wl_to_x(
            center,
            histogram.min(),
            histogram.max(),
            super::super::layout::overlay_display::canvas_coord(canvas_left),
            super::super::layout::overlay_display::canvas_coord(canvas_right),
        )),
        "window centre",
    )
    .expect("window centre fits");
    assert_eq!(
        pixel_at(center_x, strip_top + 5),
        packed([255, 190, 60, 255]),
        "histogram centre line must mark the window centre"
    );
    // One bar outside the window band renders the exact bar grey.
    let (band_left, band_right) = super::super::layout::overlay_display::window_band_x(
        center,
        width_value,
        histogram.min(),
        histogram.max(),
        super::super::layout::overlay_display::canvas_coord(canvas_left),
        super::super::layout::overlay_display::canvas_coord(canvas_right),
    );
    let canvas_span = f64::from(canvas_right - canvas_left);
    let bins = f64::from(u32::try_from(histogram.bins).expect("bin count fits"));
    let bars_bottom =
        strip_top + super::super::layout::overlay_display::HISTOGRAM_STRIP_HEIGHT - 20;
    let mut marked = false;
    for (bin, &count) in histogram.counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let bin_f = f64::from(u32::try_from(bin).expect("bin fits"));
        let x0 = super::super::layout::screen_coordinate(
            f64::from(canvas_left) + bin_f * canvas_span / bins,
            "histogram bar",
        )
        .expect("bar fits");
        let x0_f = f64::from(x0);
        if x0_f < f64::from(band_left) || x0_f > f64::from(band_right) {
            assert_eq!(
                pixel_at(x0, bars_bottom - 5),
                packed([160, 160, 160, 255]),
                "bar outside the window band must show the exact bar grey"
            );
            marked = true;
            break;
        }
    }
    assert!(marked, "fixture histogram needs a bar outside the window");
}

#[test]
fn colorbar_labels_clear_orientation_and_scale_ruler_lanes() {
    let (session, _root) = session();
    let volume = session.app.loaded.as_ref().expect("study loaded");
    let clinical = ClinicalOverlay {
        direction: volume.direction,
        patient_name: volume.patient_name.as_deref(),
        patient_id: volume.patient_id.as_deref(),
        series_description: volume.series_description.as_deref(),
        modality: volume.modality.as_deref(),
        study_date: volume.study_date.as_deref(),
        spacing: volume.spacing,
        zoom: session.app.zoom,
        cursor_value: None,
        anonymize: false,
    };
    let clinical_commands = application_overlay(
        &session.views,
        &session.viewports,
        false,
        30.0,
        Some(&clinical),
    )
    .expect("compose clinical lanes")
    .into_display_list();
    let (center, width) = session.app.browser_window_level_values();
    let mut colorbar = OverlayCommands::with_capacity(70).expect("reserve colorbar lanes");
    append_colorbar_overlay(
        &mut colorbar,
        &session.viewports[0],
        session.app.colormap,
        center,
        width,
    )
    .expect("compose colorbar labels");
    let colorbar_commands = colorbar.into_display_list();
    let half = width / 2.0;
    let labels = [
        super::super::layout::overlay_display::intensity_label(center + half),
        super::super::layout::overlay_display::intensity_label(center),
        super::super::layout::overlay_display::intensity_label(center - half),
    ];
    let label_bounds: Vec<_> = colorbar_commands
        .commands()
        .iter()
        .filter_map(|command| match command {
            DisplayCommand::DrawText { text, x, y, style }
                if labels.iter().any(|label| label == text.as_ref()) =>
            {
                style.extent(*x, *y, text)
            }
            _ => None,
        })
        .collect();
    assert_eq!(label_bounds.len(), labels.len());
    let orientation_bounds: Vec<_> = clinical_commands
        .commands()
        .iter()
        .filter_map(|command| match command {
            DisplayCommand::DrawText { text, x, y, style }
                if ["R", "L", "A", "P", "S", "I"].contains(&text.as_ref()) =>
            {
                style.extent(*x, *y, text)
            }
            _ => None,
        })
        .collect();
    let viewport = session.viewports[0];
    let image_mid = super::super::layout::screen_coordinate(
        viewport.image.x + viewport.image.width / 2.0,
        "ruler lane midpoint",
    )
    .expect("ruler midpoint fits");
    let image_left = super::super::layout::screen_coordinate(viewport.image.x, "ruler lane left")
        .expect("ruler left fits");
    let scale_bars: Vec<_> = clinical_commands
        .commands()
        .iter()
        .filter_map(|command| match command {
            DisplayCommand::FillRect { rect, radius, .. }
                if *radius == CornerRadius::SQUARE
                    && rect.height == 3
                    && rect.width >= 24
                    && rect.x >= image_left
                    && rect.x + rect.width <= image_mid =>
            {
                Some(*rect)
            }
            _ => None,
        })
        .collect();
    let scale_labels: Vec<_> = clinical_commands
        .commands()
        .iter()
        .filter_map(|command| match command {
            DisplayCommand::DrawText { text, x, y, style } if text.as_ref().ends_with(" mm") => {
                style.extent(*x, *y, text)
            }
            _ => None,
        })
        .collect();
    assert!(!scale_bars.is_empty(), "fixture must draw a scale ruler");
    let view = &session.views[0];
    let spacing = view.frame.display_spacing().values();
    let px_per_unit_x = viewport.image.width / f64::from(view.frame.width());
    let mm_per_px = spacing[1] / px_per_unit_x;
    let ruler_margin = scale_bars[0].x - image_left;
    let max_px = f64::from(image_mid - image_left - 2 * ruler_margin);
    let (_, expected_bar_px) = super::super::layout::scale_bar_for_scale(1.0 / mm_per_px, max_px)
        .expect("fixture has a measurable ruler");
    let expected_bar_width =
        super::super::layout::screen_coordinate(expected_bar_px, "expected ruler width")
            .expect("expected ruler width fits");
    assert!(scale_bars.iter().any(|bar| bar.width == expected_bar_width));
    assert!(scale_bars.iter().all(|bar| {
        scale_labels.iter().any(|label| {
            label.x < bar.x + bar.width
                && bar.x < label.x + label.width
                && label.y + label.height <= bar.y
                && bar.y - (label.y + label.height) <= bar.height * 2
        })
    }));
    let overlaps = |left: metis_platform::Rect, right: metis_platform::Rect| {
        left.x < right.x + right.width
            && right.x < left.x + left.width
            && left.y < right.y + right.height
            && right.y < left.y + left.height
    };
    for label in &label_bounds {
        assert!(
            orientation_bounds
                .iter()
                .all(|orientation| !overlaps(*label, *orientation)),
            "colorbar labels must clear orientation lanes"
        );
        assert!(
            scale_bars.iter().all(|bar| !overlaps(*label, *bar)),
            "colorbar labels must clear the scale-ruler lane"
        );
        assert!(
            scale_labels.iter().all(|ruler| !overlaps(*label, *ruler)),
            "colorbar labels must clear the scale-ruler text"
        );
    }
}
