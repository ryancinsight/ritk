//! RITK projection rendering for the optional native Métis layout.

use crate::app::SnapApp;
use crate::presentation::{PresentationFrame, PresentationSpacing};
use crate::render::{
    map_scalar_value, render_mip_axial_rgba_into, GrayscalePresentation, ProjectionStatistic,
    SlabProjection,
};
use anyhow::{anyhow, bail, Context, Result};

#[cfg(test)]
use crate::render::render_mip_axial_rgba;

use super::frame::window_level_for_app;

/// A display-only projection that shares the RITK display policy.
#[derive(Debug, Clone)]
pub(super) struct RenderedProjection {
    pub(super) frame: PresentationFrame,
    pub(super) statistic: ProjectionStatistic,
}

/// Caller-owned storage for scalar projection reduction and display mapping.
#[derive(Debug, Default)]
pub(super) struct ProjectionRenderScratch {
    pub(super) pixels: Vec<f32>,
    pub(super) rgba: Vec<u8>,
}

pub(super) fn empty_projection(statistic: ProjectionStatistic) -> Result<RenderedProjection> {
    let frame = PresentationFrame::from_rgba_storage(1, 1, vec![0, 0, 0, 255])
        .context("construct empty native projection selection frame")?;
    Ok(RenderedProjection { frame, statistic })
}

/// Render one scalar axial projection for the native Métis layout.
#[cfg(test)]
pub(super) fn render_projection(
    app: &SnapApp,
    statistic: ProjectionStatistic,
) -> Result<RenderedProjection> {
    let mut projection = empty_projection(statistic)?;
    let mut scratch = ProjectionRenderScratch::default();
    render_projection_into(app, statistic, &mut projection, &mut scratch)?;
    Ok(projection)
}

/// Re-render a scalar projection into retained frame and scratch storage.
pub(super) fn render_projection_into(
    app: &SnapApp,
    statistic: ProjectionStatistic,
    projection: &mut RenderedProjection,
    scratch: &mut ProjectionRenderScratch,
) -> Result<()> {
    let volume = app
        .loaded
        .as_ref()
        .ok_or_else(|| anyhow!("native viewer has no loaded RITK volume"))?;
    if volume.channels != 1 {
        bail!("native Métis projection requires a scalar RITK volume");
    }
    let window_level = window_level_for_app(app);
    let [width, height] = if statistic == ProjectionStatistic::Maximum {
        render_mip_axial_rgba_into(&mut scratch.rgba, volume, window_level, app.colormap)
    } else {
        render_slab_projection_into(volume, window_level, app.colormap, statistic, scratch)?
    };
    let width = u32::try_from(width).map_err(|_| anyhow!("native projection width exceeds u32"))?;
    let height =
        u32::try_from(height).map_err(|_| anyhow!("native projection height exceeds u32"))?;
    let [_, row_spacing, column_spacing] = volume.spacing;
    let spacing = PresentationSpacing::try_new(row_spacing, column_spacing)
        .context("validate native projection display spacing")?;
    projection
        .frame
        .replace_rgba_storage(width, height, spacing, &mut scratch.rgba)
        .context("replace native projection presentation storage")?;
    projection.statistic = statistic;
    Ok(())
}

fn render_slab_projection_into(
    volume: &crate::LoadedVolume,
    window_level: crate::render::WindowLevel,
    colormap: crate::render::NamedColorMap,
    statistic: ProjectionStatistic,
    scratch: &mut ProjectionRenderScratch,
) -> Result<[usize; 2]> {
    let [depth, _, _] = volume.shape;
    let center = depth
        .checked_sub(1)
        .ok_or_else(|| anyhow!("native projection volume has no depth samples"))?
        / 2;
    let request = SlabProjection::try_new(volume, 0, center, center)
        .map_err(|error| anyhow!("validate native slab projection: {error}"))?;
    let dimensions = request
        .compute_into(volume, statistic, &mut scratch.pixels)
        .map_err(|error| anyhow!("compute native slab projection: {error}"))?;
    let presentation = GrayscalePresentation::for_volume(volume)
        .map_err(|error| anyhow!("validate native projection grayscale metadata: {error}"))?;
    let byte_len = scratch
        .pixels
        .len()
        .checked_mul(4)
        .ok_or_else(|| anyhow!("native projection byte count overflows usize"))?;
    scratch.rgba.resize(byte_len, 0);
    for (pixel, &value) in scratch.rgba.chunks_exact_mut(4).zip(&scratch.pixels) {
        pixel.copy_from_slice(&map_scalar_value(
            value,
            presentation,
            window_level,
            colormap,
        ));
    }
    Ok(dimensions)
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

    #[test]
    fn native_projection_reuses_frame_and_scratch_storage_after_warmup() {
        let volume = scalar_volume();
        let mut app = SnapApp::default();
        app.load_volume(volume, "projection fixture".to_owned());
        let statistic = ProjectionStatistic::Maximum;
        let expected = render_mip_axial_rgba(
            app.loaded.as_ref().expect("loaded volume"),
            window_level_for_app(&app),
            app.colormap,
        );
        let (_, expected_rgba) = expected.into_parts();
        let mut scratch = ProjectionRenderScratch::default();
        let mut projection = empty_projection(statistic).expect("empty projection");
        render_projection_into(&app, statistic, &mut projection, &mut scratch)
            .expect("first reusable projection");
        render_projection_into(&app, statistic, &mut projection, &mut scratch)
            .expect("warm reusable projection");
        assert_eq!(projection.frame.rgba(), expected_rgba.as_ref());
        let frame_pointer = projection.frame.rgba().as_ptr();
        let scratch_pointer = scratch.rgba.as_ptr();

        render_projection_into(&app, statistic, &mut projection, &mut scratch)
            .expect("steady-state reusable projection");
        assert_eq!(projection.frame.rgba(), expected_rgba.as_ref());
        assert_eq!(projection.frame.rgba().as_ptr(), scratch_pointer);
        assert_eq!(scratch.rgba.as_ptr(), frame_pointer);
    }

    #[test]
    fn native_projection_reuses_storage_when_statistic_changes() {
        let volume = scalar_volume();
        let mut app = SnapApp::default();
        app.load_volume(volume, "projection fixture".to_owned());
        let mut scratch = ProjectionRenderScratch::default();
        let mut projection =
            empty_projection(ProjectionStatistic::Maximum).expect("empty projection");
        for statistic in [
            ProjectionStatistic::Maximum,
            ProjectionStatistic::Minimum,
            ProjectionStatistic::Average,
        ] {
            render_projection_into(&app, statistic, &mut projection, &mut scratch)
                .expect("re-render projection statistic");
            assert_eq!(projection.statistic, statistic);
            assert_eq!(
                [projection.frame.width(), projection.frame.height()],
                [2, 1]
            );
            assert_eq!(projection.frame.display_spacing().values(), [1.5, 0.75]);
        }
    }
}
