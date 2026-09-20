//! RITK projection rendering for the optional native Métis layout.

use crate::app::SnapApp;
use crate::presentation::{PresentationFrame, PresentationSpacing};
use crate::render::render_mip_axial_rgba;
use anyhow::{anyhow, bail, Context, Result};

use super::frame::window_level_for_app;

/// A display-only projection that shares the RITK display policy.
#[derive(Debug, Clone)]
pub(super) struct RenderedProjection {
    pub(super) frame: PresentationFrame,
}

pub(super) fn empty_projection() -> Result<RenderedProjection> {
    let frame = PresentationFrame::from_rgba_storage(1, 1, vec![0, 0, 0, 255])
        .context("construct empty native MIP selection frame")?;
    Ok(RenderedProjection { frame })
}

/// Render the existing RITK axial MIP for the native Métis layout.
pub(super) fn render_mip_projection(app: &SnapApp) -> Result<RenderedProjection> {
    let volume = app
        .loaded
        .as_ref()
        .ok_or_else(|| anyhow!("native viewer has no loaded RITK volume"))?;
    if volume.channels != 1 {
        bail!("native Métis MIP requires a scalar RITK volume");
    }
    let window_level = window_level_for_app(app);
    let image = render_mip_axial_rgba(volume, window_level, app.colormap);
    let ([width, height], rgba) = image.into_parts();
    let width = u32::try_from(width).map_err(|_| anyhow!("native MIP width exceeds u32"))?;
    let height = u32::try_from(height).map_err(|_| anyhow!("native MIP height exceeds u32"))?;
    let [_, row_spacing, column_spacing] = volume.spacing;
    let spacing = PresentationSpacing::try_new(row_spacing, column_spacing)
        .context("validate native MIP display spacing")?;
    let frame = PresentationFrame::from_rgba_storage(width, height, rgba.into_vec())
        .context("validate native MIP presentation frame")?
        .with_display_spacing(spacing);
    Ok(RenderedProjection { frame })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrayvec::ArrayString;
    use std::sync::Arc;

    fn scalar_volume() -> crate::LoadedVolume {
        crate::LoadedVolume {
            data: Arc::new(vec![0.0, 50.0, 100.0, 200.0]),
            shape: [2, 1, 2],
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
        let actual = render_mip_projection(&app).expect("native MIP");
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
        let error = render_mip_projection(&app).expect_err("color MIP");
        assert!(error.to_string().contains("scalar"));
    }
}
