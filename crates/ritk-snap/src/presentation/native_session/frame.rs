//! Orthogonal slice rendering for the native viewer session.

use crate::app::SnapApp;
use crate::presentation::{PresentationFrame, PresentationSpacing};
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

pub(super) fn empty_orthogonal_views() -> Result<[RenderedView; 3]> {
    let frame = PresentationFrame::from_rgba_storage(1, 1, vec![0, 0, 0, 255].into_boxed_slice())
        .context("construct empty native selection frame")?;
    Ok([
        empty_view(frame.clone(), 0, "Axial"),
        empty_view(frame.clone(), 1, "Coronal"),
        empty_view(frame, 2, "Sagittal"),
    ])
}

fn empty_view(frame: PresentationFrame, axis: usize, plane_name: &'static str) -> RenderedView {
    RenderedView {
        axis,
        plane_name,
        slice_index: 0,
        slice_count: 1,
        window_level: WindowLevel::new(
            f64::from(DEFAULT_WINDOW_CENTER),
            f64::from(DEFAULT_WINDOW_WIDTH),
        ),
        frame,
        source_size: [1, 1],
        transform: ViewTransform::default(),
    }
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
    let transform = app.view_transform;
    let display_spacing = transformed_display_spacing(frame.display_spacing(), transform);
    let (width, height, rgba) = frame.into_rgba_parts();
    let source_size = [
        usize::try_from(width).map_err(|_| anyhow!("native source width exceeds usize"))?,
        usize::try_from(height).map_err(|_| anyhow!("native source height exceeds usize"))?,
    ];
    let (output_size, rgba) = apply_to_rgba(source_size, rgba, transform)
        .context("apply RITK viewport orientation to RGBA storage")?;
    let output_width =
        u32::try_from(output_size[0]).map_err(|_| anyhow!("native frame width exceeds u32"))?;
    let output_height =
        u32::try_from(output_size[1]).map_err(|_| anyhow!("native frame height exceeds u32"))?;
    let frame = PresentationFrame::from_rgba_storage(output_width, output_height, rgba)
        .context("validate transformed RITK presentation frame")?
        .with_display_spacing(display_spacing);
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

fn transformed_display_spacing(
    spacing: PresentationSpacing,
    transform: ViewTransform,
) -> PresentationSpacing {
    match transform.rotation {
        RotationSteps::Ninety | RotationSteps::TwoSeventy => spacing.swapped(),
        RotationSteps::Zero | RotationSteps::OneEighty => spacing,
    }
}

#[cfg(test)]
mod tests {
    use super::transformed_display_spacing;
    use crate::presentation::PresentationSpacing;
    use crate::ui::{RotationSteps, ViewTransform};

    fn assert_spacing(actual: [f64; 2], expected: [f64; 2]) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            let bound = 2.0 * f64::EPSILON * expected.abs().max(1.0);
            assert!((actual - expected).abs() <= bound);
        }
    }

    #[test]
    fn transformed_spacing_follows_pixel_rotation() {
        let spacing = PresentationSpacing::try_new(2.0, 5.0).expect("spacing");
        assert_spacing(
            transformed_display_spacing(spacing, ViewTransform::default()).values(),
            [2.0, 5.0],
        );
        for rotation in [RotationSteps::Ninety, RotationSteps::TwoSeventy] {
            assert_spacing(
                transformed_display_spacing(
                    spacing,
                    ViewTransform {
                        rotation,
                        ..ViewTransform::default()
                    },
                )
                .values(),
                [5.0, 2.0],
            );
        }
    }
}
