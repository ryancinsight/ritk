//! Generate a synthetic study and capture the viewer's decoded slice images.

use anyhow::{ensure, Context, Result};
use ritk_io::{literal_arraystring, DicomReadMetadata};
use ritk_snap::dicom::loader::{load_dicom_series_from_named_bytes, load_volume_from_path};
use ritk_snap::geometry::affine::AffineTransform;
use ritk_snap::render::{
    render_fused_slice, FusedSliceParams, NamedColorMap, SliceRenderer, WindowLevel,
};
use ritk_snap::ui::voxel_to_lps;
use std::path::PathBuf;

#[path = "../src/dicom/loader/tests/fixtures.rs"]
mod fixtures;

fn main() -> Result<()> {
    let output = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .context("expected output directory")?,
    );
    let study = output.join("study");
    let files = fixtures::write_study(&study, "CT", fixtures::SERIES_UID)?;
    let volume = load_volume_from_path(&study)?;
    let borrowed: Vec<_> = files
        .iter()
        .map(|(name, bytes)| (name.clone(), bytes.as_slice()))
        .collect();
    let dropped = load_dicom_series_from_named_bytes(&borrowed)?;
    let expected: Vec<_> = fixtures::SAMPLES
        .iter()
        .map(|sample| 2.0 * f32::from(*sample) - 20.0)
        .collect();
    ensure!(
        *volume.data == expected && volume.data == dropped.data,
        "filesystem and byte-batch voxel oracle"
    );
    ensure!(
        volume.shape == fixtures::SHAPE && volume.spacing == fixtures::SPACING,
        "shape and spacing oracle"
    );
    ensure!(
        volume.origin == fixtures::ORIGIN && volume.direction == fixtures::DIRECTION,
        "physical geometry oracle"
    );
    ensure!(
        volume
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.series_instance_uid.as_deref())
            == Some(fixtures::SERIES_UID),
        "series identity oracle"
    );
    let landmark = voxel_to_lps([2, 1, 3], volume.origin, volume.direction, volume.spacing);
    ensure!(landmark == [14.0, 21.5, 31.5], "physical landmark oracle");
    let mut captures = Vec::new();
    // These parameters invert the fixture rescale under the current renderer's
    // LINEAR_EXACT equation. Default DICOM LINEAR conformance is a separate gate.
    for (axis, index, name) in [(0, 1, "depth"), (1, 1, "row"), (2, 2, "column")] {
        let rendered = SliceRenderer::render(
            &volume,
            axis,
            index,
            WindowLevel::new(235.0, 510.0),
            NamedColorMap::Grayscale,
        );
        let rgba: Vec<_> = rendered
            .pixels
            .iter()
            .flat_map(|pixel| pixel.to_array())
            .collect();
        let width = u32::try_from(rendered.size[0])?;
        let height = u32::try_from(rendered.size[1])?;
        let pixels =
            image::RgbaImage::from_raw(width, height, rgba).context("rendered RGBA dimensions")?;
        pixels.save(output.join(format!("{name}.png")))?;
        // Integer nearest-neighbour enlargement exposes the actual pixel grid;
        // it is explicitly not a physical-aspect-ratio viewport screenshot.
        image::imageops::resize(
            &pixels,
            width * 64,
            height * 64,
            image::imageops::FilterType::Nearest,
        )
        .save(output.join(format!("{name}-grid.png")))?;
        captures.push(serde_json::json!({"axis": axis, "index": index, "image": format!("{name}.png"), "size": rendered.size}));
    }

    let mut fusion_primary = volume.clone();
    let mut fusion_secondary = volume.clone();
    let frame_uid = literal_arraystring::<64>("2.25.20260905099");
    for candidate in [&mut fusion_primary, &mut fusion_secondary] {
        let metadata = candidate
            .metadata
            .get_or_insert_with(|| Box::new(DicomReadMetadata::default()));
        metadata.frame_of_reference_uid = Some(frame_uid);
    }
    // Shift the secondary grid by one column spacing in patient space. The
    // samples are deliberately distinct so normalized raster scaling and
    // physical mapping produce different pixels.
    fusion_secondary.origin[1] -= fusion_secondary.spacing[2];
    fusion_secondary.data = std::sync::Arc::new(
        fusion_secondary
            .data
            .iter()
            .map(|value| 440.0 - *value)
            .collect(),
    );
    let fusion_wl = WindowLevel::new(235.0, 510.0);
    let secondary_slice =
        ritk_snap::render::secondary_slice_for_primary(&fusion_primary, 0, 1, &fusion_secondary, 0)
            .map_err(|error| anyhow::anyhow!("derive fused slice: {error}"))?;
    let fused = render_fused_slice(
        FusedSliceParams {
            volume: &fusion_primary,
            axis: 0,
            slice: 1,
            wl: fusion_wl,
            colormap: NamedColorMap::Grayscale,
        },
        FusedSliceParams {
            volume: &fusion_secondary,
            axis: 0,
            slice: secondary_slice,
            wl: fusion_wl,
            colormap: NamedColorMap::Hot,
        },
        0.5,
    )
    .map_err(|error| anyhow::anyhow!("render fused slice: {error}"))?;
    let fused_rgba: Vec<_> = fused
        .pixels
        .iter()
        .flat_map(|pixel| pixel.to_array())
        .collect();
    let fused_width = u32::try_from(fused.size[0])?;
    let fused_height = u32::try_from(fused.size[1])?;
    let fused_pixels = image::RgbaImage::from_raw(fused_width, fused_height, fused_rgba)
        .context("fused RGBA dimensions")?;
    fused_pixels.save(output.join("fusion.png"))?;
    image::imageops::resize(
        &fused_pixels,
        fused_width * 64,
        fused_height * 64,
        image::imageops::FilterType::Nearest,
    )
    .save(output.join("fusion-grid.png"))?;
    let primary_affine = AffineTransform::from_parts(
        fusion_primary.origin,
        fusion_primary.direction,
        fusion_primary.spacing,
    )
    .map_err(|error| anyhow::anyhow!("primary fusion affine: {error}"))?;
    let secondary_affine = AffineTransform::from_parts(
        fusion_secondary.origin,
        fusion_secondary.direction,
        fusion_secondary.spacing,
    )
    .map_err(|error| anyhow::anyhow!("secondary fusion affine: {error}"))?;
    let fusion_landmarks: Vec<_> = [[1, 0, 0], [1, 0, 1], [1, 1, 2], [1, 1, 3]]
        .into_iter()
        .map(|[depth, row, col]| {
            let primary_voxel = [depth as f64, row as f64, col as f64];
            let patient = primary_affine.voxel_to_patient(primary_voxel);
            let secondary_voxel = secondary_affine.patient_to_voxel(patient);
            serde_json::json!({
                "primary_voxel": [depth, row, col],
                "patient_lps_mm": patient,
                "secondary_continuous_voxel": secondary_voxel,
                "secondary_nearest_column": secondary_voxel[2].round(),
            })
        })
        .collect();
    captures.push(serde_json::json!({
        "axis": 0,
        "index": 1,
        "secondary_index": secondary_slice,
        "image": "fusion.png",
        "size": fused.size,
        "coordinate_labels": fusion_landmarks,
    }));
    let report = serde_json::json!({
        "schema": 1, "study": "synthetic unsigned single-frame Part 10",
        "series_uid": fixtures::SERIES_UID, "shape": volume.shape,
        "spacing_mm": volume.spacing, "origin_lps_mm": volume.origin,
        "direction": volume.direction, "voxels": volume.data.as_ref(),
        "landmark_index": [2, 1, 3], "landmark_lps_mm": landmark, "captures": captures,
        "rendering": "current LINEAR_EXACT equation; software slice buffers before texture upload",
        "fusion": {
            "frame_of_reference_uid": frame_uid.as_str(),
            "secondary_origin_lps_mm": fusion_secondary.origin,
            "secondary_slice": secondary_slice,
            "sampling": "patient-coordinate nearest-neighbour with primary out-of-field retention"
        }
    });
    std::fs::write(
        output.join("workflow.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}
