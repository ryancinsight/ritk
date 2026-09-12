//! Orthogonal slice rendering for the native viewer session.

use crate::app::SnapApp;
use crate::presentation::PresentationFrame;
use crate::render::WindowLevel;
use crate::ui::{apply_to_rgba, RotationSteps, ViewTransform};
use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};
use anyhow::{anyhow, bail, Context, Result};

/// One decoded and transformed orthogonal RITK view.
#[derive(Debug, Clone)]
pub(super) struct RenderedView {
    pub(super) axis: usize,
    pub(super) plane_name: &'static str,
    pub(super) slice_index: usize,
    pub(super) slice_count: usize,
    pub(super) window_level: WindowLevel,
    pub(super) frame: PresentationFrame,
    pub(super) source_size: [usize; 2],
    pub(super) transform: ViewTransform,
    pub(super) display_spacing: [f64; 2],
}

impl RenderedView {
    pub(super) const fn frame(&self) -> &PresentationFrame {
        &self.frame
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
    let window_level = window_level_for_app(app);
    let frame = PresentationFrame::from_slice(volume, axis, index, window_level, app.colormap)
        .context("render RITK slice into a bounded presentation frame")?;
    let (width, height, rgba) = frame.into_rgba_parts();
    let source_size = [
        usize::try_from(width).map_err(|_| anyhow!("native source width exceeds usize"))?,
        usize::try_from(height).map_err(|_| anyhow!("native source height exceeds usize"))?,
    ];
    let transform = app.view_transform;
    let display_spacing = display_spacing(volume.spacing, axis, transform)?;
    let (output_size, rgba) = apply_to_rgba(source_size, rgba, transform)
        .context("apply RITK viewport orientation to RGBA storage")?;
    let output_width =
        u32::try_from(output_size[0]).map_err(|_| anyhow!("native frame width exceeds u32"))?;
    let output_height =
        u32::try_from(output_size[1]).map_err(|_| anyhow!("native frame height exceeds u32"))?;
    let frame = PresentationFrame::from_rgba_storage(output_width, output_height, rgba)
        .context("validate transformed RITK presentation frame")?;
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
        plane_name: crate::ui::anatomical_label_for_axis(Some(volume), axis),
        slice_index: index,
        slice_count: app.axis_slice_info(axis).1,
        window_level,
        frame,
        source_size,
        transform,
        display_spacing,
    })
}

pub(super) fn window_level_for_app(app: &SnapApp) -> WindowLevel {
    WindowLevel::new(
        app.viewer_state
            .window_center
            .map_or(f64::from(DEFAULT_WINDOW_CENTER), f64::from),
        app.viewer_state
            .window_width
            .map_or(f64::from(DEFAULT_WINDOW_WIDTH), f64::from)
            .max(1.0),
    )
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
