use super::*;
use crate::render::WindowLevel;
use crate::ui::ViewTransform;

const SURFACE_WIDTH: u32 = 80;
const SURFACE_HEIGHT: u32 = 60;
const PANEL_WIDTH: u32 = (SURFACE_WIDTH - VIEW_GAP_PIXELS) / 2;
const PANEL_HEIGHT: u32 = (SURFACE_HEIGHT - VIEW_GAP_PIXELS) / 2;

fn view(axis: usize, color: [u8; 4]) -> RenderedView {
    RenderedView {
        axis,
        plane_name: ["Axial", "Coronal", "Sagittal"][axis],
        slice_index: 0,
        slice_count: 1,
        window_level: WindowLevel::new(0.0, 1.0),
        frame: PresentationFrame::from_rgba(1, 1, &color).expect("one-pixel presentation frame"),
        source_size: [1, 1],
        transform: ViewTransform::default(),
    }
}

fn views() -> [RenderedView; 3] {
    [
        view(0, [255, 0, 0, 255]),
        view(1, [0, 255, 0, 255]),
        view(2, [0, 0, 255, 255]),
    ]
}

fn navigation() -> PaneNavigation {
    PaneNavigation {
        zoom: 1.0,
        pan_offset: ViewportOffset::new(0.0, 0.0),
    }
}

fn compose(
    views: &[RenderedView; 3],
    fourth_frame: &PresentationFrame,
    surface_size: [u32; 2],
) -> FourPanelComposition {
    compose_four_panel(
        views,
        fourth_frame,
        surface_size,
        ViewportArea::full(surface_size[0], surface_size[1]),
        navigation(),
        navigation(),
    )
    .expect("valid four-panel composition")
}

fn panel_bounds(viewport: NativeViewport) -> PanelBounds {
    PanelBounds {
        x: viewport.panel_x,
        y: viewport.panel_y,
        width: viewport.panel_width,
        height: viewport.panel_height,
    }
}

fn panels(composition: &FourPanelComposition) -> [PanelBounds; 4] {
    [
        panel_bounds(composition.viewports[0]),
        panel_bounds(composition.viewports[1]),
        panel_bounds(composition.viewports[2]),
        composition.fourth_panel,
    ]
}

fn assert_layout(surface_size: [u32; 2], actual: [PanelBounds; 4]) {
    let [width, height] = surface_size;
    let available_width = width - VIEW_GAP_PIXELS;
    let available_height = height - VIEW_GAP_PIXELS;
    let left_width = available_width / 2 + available_width % 2;
    let right_width = available_width / 2;
    let top_height = available_height / 2 + available_height % 2;
    let bottom_height = available_height / 2;
    let expected = [
        PanelBounds {
            x: 0,
            y: 0,
            width: left_width,
            height: top_height,
        },
        PanelBounds {
            x: left_width + VIEW_GAP_PIXELS,
            y: 0,
            width: right_width,
            height: top_height,
        },
        PanelBounds {
            x: 0,
            y: top_height + VIEW_GAP_PIXELS,
            width: left_width,
            height: bottom_height,
        },
        PanelBounds {
            x: left_width + VIEW_GAP_PIXELS,
            y: top_height + VIEW_GAP_PIXELS,
            width: right_width,
            height: bottom_height,
        },
    ];
    assert_eq!(actual, expected);
    for panel in actual {
        assert!(panel.x + panel.width <= width);
        assert!(panel.y + panel.height <= height);
    }
    for first in 0..actual.len() {
        for second in first + 1..actual.len() {
            let a = actual[first];
            let b = actual[second];
            assert!(
                a.x + a.width <= b.x
                    || b.x + b.width <= a.x
                    || a.y + a.height <= b.y
                    || b.y + b.height <= a.y,
                "panels {a:?} and {b:?} overlap"
            );
        }
    }
    assert_eq!(
        actual[1].x - (actual[0].x + actual[0].width),
        VIEW_GAP_PIXELS
    );
    assert_eq!(
        actual[3].x - (actual[2].x + actual[2].width),
        VIEW_GAP_PIXELS
    );
    assert_eq!(
        actual[2].y - (actual[0].y + actual[0].height),
        VIEW_GAP_PIXELS
    );
    assert_eq!(
        actual[3].y - (actual[1].y + actual[1].height),
        VIEW_GAP_PIXELS
    );
}

fn panel_contains(panel: PanelBounds, x: u32, y: u32) -> bool {
    x >= panel.x && x < panel.x + panel.width && y >= panel.y && y < panel.y + panel.height
}

#[test]
fn four_panel_composition_places_each_frame_inside_its_bounds() {
    let views = views();
    let fourth =
        PresentationFrame::from_rgba(1, 1, &[255, 255, 255, 255]).expect("one-pixel fourth frame");
    for surface_size in [[SURFACE_WIDTH, SURFACE_HEIGHT], [73, 107], [119, 83]] {
        let composition = compose(&views, &fourth, surface_size);
        let panels = panels(&composition);
        assert_layout(surface_size, panels);
        for (view, viewport) in views.iter().zip(composition.viewports) {
            let (x, y) = viewport.center();
            let rgba = view.frame.rgba();
            assert_eq!(
                composition.framebuffer.get_pixel(x, y),
                Color::rgba(rgba[0], rgba[1], rgba[2], rgba[3])
            );
        }
        let [width, height] = surface_size;
        for y in 0..height {
            for x in 0..width {
                if !panels.iter().any(|panel| panel_contains(*panel, x, y)) {
                    assert_eq!(
                        composition.framebuffer.get_pixel(
                            i32::try_from(x).expect("test x fits host coordinate"),
                            i32::try_from(y).expect("test y fits host coordinate"),
                        ),
                        Color::BLACK
                    );
                }
            }
        }
    }
}

#[test]
fn fourth_panel_preserves_every_rgba_pixel_of_the_physical_frame() {
    let views = views();
    let mut rgba = Vec::with_capacity(
        usize::try_from(PANEL_WIDTH * PANEL_HEIGHT * 4).expect("fixture byte count fits usize"),
    );
    for y in 0..PANEL_HEIGHT {
        for x in 0..PANEL_WIDTH {
            rgba.extend_from_slice(&[
                u8::try_from(x).expect("fixture x fits u8"),
                u8::try_from(y).expect("fixture y fits u8"),
                u8::try_from(x + y).expect("fixture coordinate sum fits u8"),
                255,
            ]);
        }
    }
    let fourth = PresentationFrame::from_rgba(PANEL_WIDTH, PANEL_HEIGHT, &rgba)
        .expect("patterned physical frame");
    let composition = compose(&views, &fourth, [SURFACE_WIDTH, SURFACE_HEIGHT]);
    assert_eq!(
        composition.fourth_panel,
        PanelBounds {
            x: 42,
            y: 32,
            width: PANEL_WIDTH,
            height: PANEL_HEIGHT,
        }
    );
    for y in 0..PANEL_HEIGHT {
        for x in 0..PANEL_WIDTH {
            let offset = usize::try_from((y * PANEL_WIDTH + x) * 4)
                .expect("fixture pixel offset fits usize");
            let expected = Color::rgba(
                rgba[offset],
                rgba[offset + 1],
                rgba[offset + 2],
                rgba[offset + 3],
            );
            assert_eq!(
                composition.framebuffer.get_pixel(
                    i32::try_from(composition.fourth_panel.x + x)
                        .expect("test x fits host coordinate"),
                    i32::try_from(composition.fourth_panel.y + y)
                        .expect("test y fits host coordinate"),
                ),
                expected,
                "fourth-panel pixel at ({x}, {y})"
            );
        }
    }
}

#[test]
fn four_panel_composition_rejects_zero_and_unpartitionable_surfaces() {
    let views = views();
    let fourth =
        PresentationFrame::from_rgba(1, 1, &[255, 255, 255, 255]).expect("one-pixel fourth frame");
    for (surface_size, expected) in [
        (
            [0, 60],
            "native surface dimensions must be nonzero while rendering",
        ),
        (
            [80, 0],
            "native surface dimensions must be nonzero while rendering",
        ),
        (
            [3, 60],
            "native four-panel viewport is narrower than its separator",
        ),
        (
            [80, 3],
            "native four-panel viewport is shorter than its separator",
        ),
        (
            [5, 60],
            "native four-panel layout cannot allocate four panels",
        ),
        (
            [80, 5],
            "native four-panel layout cannot allocate four panels",
        ),
    ] {
        assert_eq!(
            compose_four_panel(
                &views,
                &fourth,
                surface_size,
                ViewportArea::full(surface_size[0], surface_size[1]),
                navigation(),
                navigation(),
            )
            .err()
            .map(|error| error.to_string())
            .as_deref(),
            Some(expected)
        );
    }
}
