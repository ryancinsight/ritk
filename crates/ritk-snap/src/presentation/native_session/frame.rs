//! Orthogonal slice composition for the native viewer session.

use crate::app::action_adapter::ViewerViewport;
use crate::app::SnapApp;
use crate::presentation::PresentationFrame;
use crate::render::{SliceRenderer, WindowLevel};
use crate::ui::{apply_to_image, RotationSteps, ViewTransform};
use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};
use anyhow::{anyhow, bail, Context, Result};
use metis_platform::{Color, Framebuffer};

/// Pixel separator between the three native orthogonal panels.
pub(super) const VIEW_GAP_PIXELS: u32 = 4;

/// One decoded and transformed orthogonal RITK view.
#[derive(Debug, Clone)]
pub(super) struct RenderedView {
    axis: usize,
    frame: PresentationFrame,
    source_size: [usize; 2],
    transform: ViewTransform,
    display_spacing: [f64; 2],
}

impl RenderedView {
    pub(super) const fn frame(&self) -> &PresentationFrame {
        &self.frame
    }
}

/// Screen placement and RITK coordinate mapping for one composed view.
#[derive(Debug, Clone, Copy)]
pub(super) struct NativeViewport {
    panel: ScreenRect,
    image: ScreenRect,
    mapping: ViewerViewport,
}

impl NativeViewport {
    pub(super) const fn axis(self) -> usize {
        self.mapping.axis()
    }

    pub(super) const fn mapping(self) -> ViewerViewport {
        self.mapping
    }

    pub(super) fn contains(self, x: f64, y: f64) -> bool {
        self.panel.contains(x, y)
    }

    #[cfg(test)]
    pub(super) fn center(self) -> (i32, i32) {
        let x = (self.image.x + self.image.width * 0.5).round();
        let y = (self.image.y + self.image.height * 0.5).round();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "test coordinates are bounded by the native surface"
        )]
        let x = x as i32;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "test coordinates are bounded by the native surface"
        )]
        let y = y as i32;
        (x, y)
    }
}

#[derive(Debug, Clone, Copy)]
struct ScreenRect {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

impl ScreenRect {
    fn contains(self, x: f64, y: f64) -> bool {
        x.is_finite()
            && y.is_finite()
            && x >= self.x
            && y >= self.y
            && x < self.x + self.width
            && y < self.y + self.height
    }
}

/// Render all three RITK orthogonal planes with shared display semantics.
pub(super) fn render_orthogonal_views(app: &SnapApp) -> Result<[RenderedView; 3]> {
    let views = [
        render_view(app, 0)?,
        render_view(app, 1)?,
        render_view(app, 2)?,
    ];
    Ok(views)
}

fn render_view(app: &SnapApp, axis: usize) -> Result<RenderedView> {
    let volume = app
        .loaded
        .as_ref()
        .ok_or_else(|| anyhow!("native viewer has no loaded RITK volume"))?;
    let (index, _) = app.axis_slice_info(axis);
    let window_level = WindowLevel::new(
        app.viewer_state
            .window_center
            .map_or(f64::from(DEFAULT_WINDOW_CENTER), f64::from),
        app.viewer_state
            .window_width
            .map_or(f64::from(DEFAULT_WINDOW_WIDTH), f64::from)
            .max(1.0),
    );
    let image = SliceRenderer::render(volume, axis, index, window_level, app.colormap);
    let source_size = image.size;
    let transform = app.view_transform;
    let display_spacing = display_spacing(volume.spacing, axis, transform)?;
    let image = apply_to_image(&image, transform);
    let frame = PresentationFrame::from_color_image(&image)
        .context("convert RITK slice to a bounded presentation frame")?;
    let output_size = transform.output_size(source_size);
    let frame_size = [
        usize::try_from(frame.width()).map_err(|_| anyhow!("native frame width exceeds usize"))?,
        usize::try_from(frame.height())
            .map_err(|_| anyhow!("native frame height exceeds usize"))?,
    ];
    if frame_size != output_size {
        bail!(
            "native frame dimensions {:?} do not match transformed slice {:?}",
            frame_size,
            output_size
        );
    }
    Ok(RenderedView {
        axis,
        frame,
        source_size,
        transform,
        display_spacing,
    })
}

fn display_spacing(spacing: [f64; 3], axis: usize, transform: ViewTransform) -> Result<[f64; 2]> {
    let [dz, dy, dx] = spacing;
    let source = match axis {
        0 => [dy, dx],
        1 => [dz, dx],
        2 => [dz, dy],
        _ => bail!("native viewer axis {axis} is outside the supported range 0..=2"),
    };
    let display = match transform.rotation {
        RotationSteps::Ninety | RotationSteps::TwoSeventy => [source[1], source[0]],
        RotationSteps::Zero | RotationSteps::OneEighty => source,
    };
    if !display
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
    {
        bail!("native viewer sample distances must be finite and positive");
    }
    Ok(display)
}

/// Compose the three views into one bounded Métis framebuffer.
pub(super) fn surface_frames(
    views: &[RenderedView; 3],
    surface_width: u32,
    surface_height: u32,
    zoom: f32,
    pan_offset: egui::Vec2,
) -> Result<(Framebuffer, [NativeViewport; 3])> {
    if surface_width == 0 || surface_height == 0 {
        bail!("native surface dimensions must be nonzero while rendering");
    }
    if !zoom.is_finite() || zoom <= 0.0 {
        bail!("native viewer zoom must be finite and positive");
    }
    let gaps = VIEW_GAP_PIXELS
        .checked_mul(2)
        .ok_or_else(|| anyhow!("native view gap arithmetic overflows"))?;
    let available_width = surface_width
        .checked_sub(gaps)
        .ok_or_else(|| anyhow!("native surface is narrower than its view separators"))?;
    if available_width < 3 {
        bail!("native surface cannot allocate three orthogonal view panels");
    }
    let base_width = available_width / 3;
    let remainder = available_width % 3;
    let panel_widths = [
        base_width + u32::from(remainder > 0),
        base_width + u32::from(remainder > 1),
        base_width,
    ];
    let mut framebuffer = Framebuffer::new(surface_width, surface_height)
        .map_err(|error| anyhow!("allocate native viewer framebuffer: {error}"))?;
    framebuffer.clear(Color::BLACK);
    let viewports = [
        placement(
            &views[0],
            0,
            panel_widths[0],
            surface_height,
            zoom,
            pan_offset,
        )?,
        placement(
            &views[1],
            panel_widths[0] + VIEW_GAP_PIXELS,
            panel_widths[1],
            surface_height,
            zoom,
            pan_offset,
        )?,
        placement(
            &views[2],
            panel_widths[0] + VIEW_GAP_PIXELS + panel_widths[1] + VIEW_GAP_PIXELS,
            panel_widths[2],
            surface_height,
            zoom,
            pan_offset,
        )?,
    ];
    for (view, viewport) in views.iter().zip(viewports) {
        blit_frame(
            &mut framebuffer,
            view,
            viewport,
            surface_width,
            surface_height,
        )?;
    }
    Ok((framebuffer, viewports))
}

fn placement(
    view: &RenderedView,
    panel_x: u32,
    panel_width: u32,
    surface_height: u32,
    zoom: f32,
    pan_offset: egui::Vec2,
) -> Result<NativeViewport> {
    let frame_width = f64::from(view.frame.width());
    let frame_height = f64::from(view.frame.height());
    let reference = view.display_spacing[0].max(view.display_spacing[1]);
    let relative_x = view.display_spacing[1] / reference;
    let relative_y = view.display_spacing[0] / reference;
    let physical_width = frame_width * relative_x;
    let physical_height = frame_height * relative_y;
    if !physical_width.is_finite()
        || !physical_height.is_finite()
        || physical_width <= 0.0
        || physical_height <= 0.0
    {
        bail!("native view physical geometry is outside the finite positive range");
    }
    let fit =
        (f64::from(panel_width) / physical_width).min(f64::from(surface_height) / physical_height);
    let zoom = f64::from(zoom);
    let texel_x = relative_x * fit * zoom;
    let texel_y = relative_y * fit * zoom;
    let rendered_width = frame_width * texel_x;
    let rendered_height = frame_height * texel_y;
    let origin_x = f64::from(panel_x)
        + (f64::from(panel_width) - rendered_width) * 0.5
        + f64::from(pan_offset.x);
    let origin_y = (f64::from(surface_height) - rendered_height) * 0.5 + f64::from(pan_offset.y);
    if ![
        texel_x,
        texel_y,
        rendered_width,
        rendered_height,
        origin_x,
        origin_y,
    ]
    .iter()
    .all(|value| value.is_finite() && *value > f64::from(f32::MIN))
        || texel_x > f64::from(f32::MAX)
        || texel_y > f64::from(f32::MAX)
        || origin_x > f64::from(f32::MAX)
        || origin_y > f64::from(f32::MAX)
    {
        bail!("native viewer geometry exceeds finite host range");
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "native geometry is checked against the f32 range above"
    )]
    let origin = egui::pos2(origin_x as f32, origin_y as f32);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "native geometry is checked against the f32 range above"
    )]
    let texel = egui::vec2(texel_x as f32, texel_y as f32);
    let mapping = ViewerViewport::new(view.axis, origin, texel, view.source_size, view.transform)
        .map_err(|error| anyhow!("construct native viewer viewport: {error}"))?;
    Ok(NativeViewport {
        panel: ScreenRect {
            x: f64::from(panel_x),
            y: 0.0,
            width: f64::from(panel_width),
            height: f64::from(surface_height),
        },
        image: ScreenRect {
            x: origin_x,
            y: origin_y,
            width: rendered_width,
            height: rendered_height,
        },
        mapping,
    })
}

fn blit_frame(
    framebuffer: &mut Framebuffer,
    view: &RenderedView,
    viewport: NativeViewport,
    surface_width: u32,
    surface_height: u32,
) -> Result<()> {
    let frame_width = f64::from(view.frame.width());
    let frame_height = f64::from(view.frame.height());
    let texel_x = viewport.image.width / frame_width;
    let texel_y = viewport.image.height / frame_height;
    let frame_width_usize = usize::try_from(view.frame.width())?;
    let frame_height_usize = usize::try_from(view.frame.height())?;
    for y in 0..surface_height {
        let screen_y = f64::from(y) + 0.5;
        let source_y = ((screen_y - viewport.image.y) / texel_y).floor();
        if source_y < 0.0 || source_y >= frame_height {
            continue;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "source coordinate is checked against the bounded frame height"
        )]
        let source_y = source_y as usize;
        for x in 0..surface_width {
            let screen_x = f64::from(x) + 0.5;
            let source_x = ((screen_x - viewport.image.x) / texel_x).floor();
            if source_x < 0.0 || source_x >= frame_width {
                continue;
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "source coordinate is checked against the bounded frame width"
            )]
            let source_x = source_x as usize;
            let offset = source_y
                .checked_mul(frame_width_usize)
                .and_then(|row| row.checked_add(source_x))
                .and_then(|pixel| pixel.checked_mul(4))
                .ok_or_else(|| anyhow!("native viewer frame offset overflows"))?;
            let pixel = view
                .frame
                .rgba()
                .get(offset..offset + 4)
                .ok_or_else(|| anyhow!("native viewer frame storage is truncated"))?;
            let x = i32::try_from(x)?;
            let y = i32::try_from(y)?;
            framebuffer.set_pixel(x, y, Color::rgba(pixel[0], pixel[1], pixel[2], pixel[3]));
        }
    }
    debug_assert_eq!(
        frame_height_usize.checked_mul(frame_width_usize),
        Some(view.frame.rgba().len() / 4)
    );
    Ok(())
}
