//! Native histogram window panel and per-pane colorbar overlays.

use super::support::session;
use iris::color::{ColorMap, Normalized};

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
        - super::super::layout::COLORBAR_WIDTH;
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
    let strip_top = panel.y + panel.height - super::super::layout::HISTOGRAM_STRIP_HEIGHT;
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
            super::super::layout::canvas_f32(canvas_left),
            super::super::layout::canvas_f32(canvas_right),
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
    let (band_left, band_right) = super::super::layout::window_band_x(
        center,
        width_value,
        histogram.min(),
        histogram.max(),
        super::super::layout::canvas_f32(canvas_left),
        super::super::layout::canvas_f32(canvas_right),
    );
    let canvas_span = f64::from(canvas_right - canvas_left);
    let bins = f64::from(u32::try_from(histogram.bins).expect("bin count fits"));
    let bars_bottom = strip_top + super::super::layout::HISTOGRAM_STRIP_HEIGHT - 20;
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
