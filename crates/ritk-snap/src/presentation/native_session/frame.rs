//! Orthogonal slice rendering for the native viewer session.

use crate::app::SnapApp;
use crate::presentation::{PresentationFrame, PresentationSpacing};
use crate::render::{FrameRenderScratch, WindowLevel};
use crate::ui::{RotationSteps, ViewTransform};
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
pub(super) fn render_orthogonal_views(
    app: &SnapApp,
    scratch: &mut [FrameRenderScratch; 3],
) -> Result<[RenderedView; 3]> {
    let mut views = empty_orthogonal_views()?;
    render_orthogonal_views_into(app, &mut views, scratch)?;
    Ok(views)
}

/// Re-render orthogonal views into retained frames and scratch storage.
pub(super) fn render_orthogonal_views_into(
    app: &SnapApp,
    views: &mut [RenderedView; 3],
    scratch: &mut [FrameRenderScratch; 3],
) -> Result<()> {
    for (axis, (view, scratch)) in views.iter_mut().zip(scratch.iter_mut()).enumerate() {
        render_view_into(app, axis, view, scratch)?;
    }
    Ok(())
}

pub(super) fn empty_orthogonal_views() -> Result<[RenderedView; 3]> {
    let frame = PresentationFrame::from_rgba_storage(1, 1, vec![0, 0, 0, 255])
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

fn render_view_into(
    app: &SnapApp,
    axis: usize,
    view: &mut RenderedView,
    scratch: &mut FrameRenderScratch,
) -> Result<()> {
    let volume = app
        .loaded
        .as_ref()
        .ok_or_else(|| anyhow!("native viewer has no loaded RITK volume"))?;
    let (index, _) = app.axis_slice_info(axis);
    let window_level = window_level_for_app(app);
    view.frame
        .render_slice_into(volume, axis, index, window_level, app.colormap, scratch)
        .context("render RITK slice into a bounded presentation frame")?;
    let transform = app.view_transform;
    let display_spacing = transformed_display_spacing(view.frame.display_spacing(), transform);
    let source_size = [
        usize::try_from(view.frame.width())
            .map_err(|_| anyhow!("native source width exceeds usize"))?,
        usize::try_from(view.frame.height())
            .map_err(|_| anyhow!("native source height exceeds usize"))?,
    ];
    view.frame
        .apply_view_transform(transform, display_spacing, scratch)
        .context("apply RITK viewport orientation to reusable RGBA storage")?;
    let output_size = transform.output_size(source_size);
    let frame_size = [
        usize::try_from(view.frame.width())
            .map_err(|_| anyhow!("native frame width exceeds usize"))?,
        usize::try_from(view.frame.height())
            .map_err(|_| anyhow!("native frame height exceeds usize"))?,
    ];
    if frame_size != output_size {
        bail!(
            "native frame dimensions {:?} do not match transformed slice {:?}",
            frame_size,
            output_size
        );
    }
    view.axis = axis;
    view.plane_name = crate::ui::anatomical_label_for_axis(Some(volume), axis);
    view.slice_index = index;
    view.slice_count = app.axis_slice_info(axis).1;
    view.window_level = window_level;
    view.source_size = source_size;
    view.transform = transform;
    Ok(())
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
