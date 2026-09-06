//! Generate a synthetic study and capture the viewer's decoded slice images.

use anyhow::{ensure, Context, Result};
use ritk_snap::dicom::loader::{load_dicom_series_from_named_bytes, load_volume_from_path};
use ritk_snap::render::{NamedColorMap, SliceRenderer, WindowLevel};
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
    let report = serde_json::json!({
        "schema": 1, "study": "synthetic unsigned single-frame Part 10",
        "series_uid": fixtures::SERIES_UID, "shape": volume.shape,
        "spacing_mm": volume.spacing, "origin_lps_mm": volume.origin,
        "direction": volume.direction, "voxels": volume.data.as_ref(),
        "landmark_index": [2, 1, 3], "landmark_lps_mm": landmark, "captures": captures,
        "rendering": "current LINEAR_EXACT equation; software slice buffers before texture upload"
    });
    std::fs::write(
        output.join("workflow.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}
