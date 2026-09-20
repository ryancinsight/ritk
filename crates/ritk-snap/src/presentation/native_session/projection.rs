//! RITK projection rendering for the optional native Métis layout.

use crate::app::SnapApp;
use crate::presentation::{PresentationFrame, PresentationSpacing};
use crate::render::{
    map_scalar_value, render_mip_axial_rgba, GrayscalePresentation, ProjectionStatistic,
    SlabProjection,
};
use anyhow::{anyhow, bail, Context, Result};

use super::frame::window_level_for_app;

/// A display-only projection that shares the RITK display policy.
#[derive(Debug, Clone)]
pub(super) struct RenderedProjection {
    pub(super) frame: PresentationFrame,
    pub(super) statistic: ProjectionStatistic,
}

pub(super) fn empty_projection(statistic: ProjectionStatistic) -> Result<RenderedProjection> {
    let frame = PresentationFrame::from_rgba_storage(1, 1, vec![0, 0, 0, 255])
        .context("construct empty native projection selection frame")?;
    Ok(RenderedProjection { frame, statistic })
}

/// Render one scalar axial projection for the native Métis layout.
pub(super) fn render_projection(
    app: &SnapApp,
    statistic: ProjectionStatistic,
) -> Result<RenderedProjection> {
    let volume = app
        .loaded
        .as_ref()
        .ok_or_else(|| anyhow!("native viewer has no loaded RITK volume"))?;
    if volume.channels != 1 {
        bail!("native Métis projection requires a scalar RITK volume");
    }
    let window_level = window_level_for_app(app);
    let image = if statistic == ProjectionStatistic::Maximum {
        render_mip_axial_rgba(volume, window_level, app.colormap)
    } else {
        render_slab_projection(volume, window_level, app.colormap, statistic)?
    };
    let ([width, height], rgba) = image.into_parts();
    let width = u32::try_from(width).map_err(|_| anyhow!("native projection width exceeds u32"))?;
    let height =
        u32::try_from(height).map_err(|_| anyhow!("native projection height exceeds u32"))?;
    let [_, row_spacing, column_spacing] = volume.spacing;
    let spacing = PresentationSpacing::try_new(row_spacing, column_spacing)
        .context("validate native projection display spacing")?;
    let frame = PresentationFrame::from_rgba_storage(width, height, rgba.into_vec())
        .context("validate native projection presentation frame")?
        .with_display_spacing(spacing);
    Ok(RenderedProjection { frame, statistic })
}

fn render_slab_projection(
    volume: &crate::LoadedVolume,
    window_level: crate::render::WindowLevel,
    colormap: crate::render::NamedColorMap,
    statistic: ProjectionStatistic,
) -> Result<crate::render::RgbaImage> {
    let [depth, _, _] = volume.shape;
    let center = depth
        .checked_sub(1)
        .ok_or_else(|| anyhow!("native projection volume has no depth samples"))?
        / 2;
    let request = SlabProjection::try_new(volume, 0, center, center)
        .map_err(|error| anyhow!("validate native slab projection: {error}"))?;
    let plane = request
        .compute(volume, statistic)
        .map_err(|error| anyhow!("compute native slab projection: {error}"))?;
    let presentation = GrayscalePresentation::for_volume(volume)
        .map_err(|error| anyhow!("validate native projection grayscale metadata: {error}"))?;
    let byte_len = plane
        .pixels()
        .len()
        .checked_mul(4)
        .ok_or_else(|| anyhow!("native projection byte count overflows usize"))?;
    let mut rgba = Vec::new();
    rgba.try_reserve_exact(byte_len)
        .map_err(|_| anyhow!("reserve native projection pixels"))?;
    for &value in plane.pixels() {
        rgba.extend_from_slice(&map_scalar_value(
            value,
            presentation,
            window_level,
            colormap,
        ));
    }
    Ok(crate::render::RgbaImage::new(plane.dimensions(), rgba))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrayvec::ArrayString;
    use std::sync::Arc;

    fn scalar_volume() -> crate::LoadedVolume {
        crate::LoadedVolume {
            data: Arc::new(vec![0.0, 50.0, 100.0, 200.0, 300.0, 400.0]),
            shape: [3, 1, 2],
            channels: 1,
            spacing: [2.0, 1.5, 0.75],
            origin: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            metadata: None,
            source: None,
            modality: Some(ArrayString::from("CT").expect("bounded modality")),
            patient_name: None,
            patient_id: None,
            study_date: None,
            series_description: None,
            series_time: None,
            patient_weight_kg: None,
            injected_dose_bq: None,
            radionuclide_half_life_s: None,
            radiopharmaceutical_start_time: None,
            decay_correction: None,
        }
    }

    #[test]
    fn native_projection_matches_the_existing_ritk_mip_renderer() {
        let volume = scalar_volume();
        let mut app = SnapApp::default();
        app.load_volume(volume.clone(), "projection fixture".to_owned());
        let window_level = window_level_for_app(&app);
        let expected = render_mip_axial_rgba(&volume, window_level, app.colormap);
        let actual = render_projection(&app, ProjectionStatistic::Maximum).expect("native MIP");
        let ([expected_width, expected_height], expected_rgba) = expected.into_parts();
        assert_eq!(
            actual.frame.width(),
            u32::try_from(expected_width).expect("width")
        );
        assert_eq!(
            actual.frame.height(),
            u32::try_from(expected_height).expect("height")
        );
        assert_eq!(actual.frame.rgba(), expected_rgba.as_ref());
        assert_eq!(actual.frame.display_spacing().values(), [1.5, 0.75]);
    }

    #[test]
    fn native_projection_rejects_color_volume() {
        let mut volume = scalar_volume();
        volume.channels = 3;
        volume.data = Arc::new(vec![0.0; 2 * 1 * 2 * 3]);
        let mut app = SnapApp::default();
        app.load_volume(volume, "color fixture".to_owned());
        let error = render_projection(&app, ProjectionStatistic::Maximum).expect_err("color MIP");
        assert!(error.to_string().contains("scalar"));
    }

    #[test]
    fn native_projection_reductions_use_the_typed_slab_contract() {
        let volume = scalar_volume();
        let mut app = SnapApp::default();
        app.load_volume(volume.clone(), "projection fixture".to_owned());
        let minimum = render_projection(&app, ProjectionStatistic::Minimum).expect("native MinIP");
        let average =
            render_projection(&app, ProjectionStatistic::Average).expect("native average");
        assert_eq!([minimum.frame.width(), minimum.frame.height()], [2, 1]);
        assert_eq!([average.frame.width(), average.frame.height()], [2, 1]);
        let request = SlabProjection::try_new(&volume, 0, 1, 1).expect("full-depth slab");
        let minimum_plane = request
            .compute(&volume, ProjectionStatistic::Minimum)
            .expect("minimum plane");
        let average_plane = request
            .compute(&volume, ProjectionStatistic::Average)
            .expect("average plane");
        assert_ne!(minimum_plane.pixels(), average_plane.pixels());
    }
}
