//! Visual comparison of CT↔MR rigid registration: identity (before), native
//! classical mutual-information registration, and the SimpleElastix reference.
//!
//! Renders a mid-axial slice as an RGB overlay (R = CT, G = MR). Aligned
//! anatomy appears yellow/grey; misalignment appears as red/green fringes.
//! The three panels are identity | RITK MI | Elastix.
//!
//! Usage: `cargo run --release -p ritk-registration --example registration_compare_figure`
//! (paths default to the RIRE-109 pair + Elastix result under leoneuro/).

use anyhow::{bail, Context, Result};
use coeus_core::SequentialBackend;
use eunomia::CastFrom;
use image::{Rgb, RgbImage};
use ritk_filter::resample::native::{fixed_world_points, resample_moving_at_world};
use ritk_image::Image;
use ritk_io::{
    format::nifti::native::{NiftiReader, NiftiWriter},
    ImageReader, ImageWriter,
};
use ritk_registration::{
    classical::{
        engine::{ClassicalConfig, MutualInformationMetric},
        image_to_leto_volume, index_affine_to_physical, ImageRegistration,
    },
    AffineTransform,
};
use ritk_transform::transform::affine::AtlasAffineTransform;

type Backend = SequentialBackend;

fn host(image: &Image<f32, Backend, 3>) -> Result<Vec<f32>> {
    Ok(image.data_slice()?.to_vec())
}

/// Normalise an axial slice `z0` of a `[nz, ny, nx]` row-major volume to
/// `[0, 1]` using a robust `[p2, p98]` window.
fn slice_norm(volume: &[f32], shape: [usize; 3], z0: usize) -> Result<Vec<f32>> {
    let [nz, ny, nx] = shape;
    if z0 >= nz {
        bail!("axial slice index {z0} exceeds volume depth {nz}");
    }
    let base = z0
        .checked_mul(ny)
        .and_then(|value| value.checked_mul(nx))
        .context("slice offset overflows usize")?;
    let count = ny
        .checked_mul(nx)
        .context("axial slice voxel count overflows usize")?;
    let slice = volume
        .get(base..base + count)
        .context("volume length does not match declared shape")?;
    let mut sorted: Vec<f32> = slice
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if sorted.is_empty() {
        bail!("cannot normalise an axial slice without finite voxels");
    }
    sorted.sort_by(f32::total_cmp);
    let percentile = |hundredths| {
        let index = sorted.len().saturating_mul(hundredths) / 100;
        sorted[index.min(sorted.len() - 1)]
    };
    let lower = percentile(2);
    let upper = percentile(98);
    let range = (upper - lower).max(f32::EPSILON);
    Ok(slice
        .iter()
        .map(|value| ((value - lower) / range).clamp(0.0, 1.0))
        .collect())
}

fn normalized_to_u8(value: f32) -> u8 {
    u8::cast_from(value.clamp(0.0, 1.0) * 255.0)
}

fn ncc(fixed: &[f32], moving: &[f32]) -> Result<f64> {
    if fixed.len() != moving.len() || fixed.is_empty() {
        bail!(
            "NCC requires equally sized non-empty volumes, got fixed={} moving={}",
            fixed.len(),
            moving.len()
        );
    }
    let count = f64::cast_from(u64::try_from(fixed.len()).context("voxel count exceeds u64")?);
    let fixed_mean = fixed.iter().copied().map(f64::from).sum::<f64>() / count;
    let moving_mean = moving.iter().copied().map(f64::from).sum::<f64>() / count;
    let (numerator, fixed_energy, moving_energy) =
        fixed.iter().copied().zip(moving.iter().copied()).fold(
            (0.0, 0.0, 0.0),
            |(numerator, fixed_energy, moving_energy), (fixed_value, moving_value)| {
                let fixed_centered = f64::from(fixed_value) - fixed_mean;
                let moving_centered = f64::from(moving_value) - moving_mean;
                (
                    numerator + fixed_centered * moving_centered,
                    fixed_energy + fixed_centered * fixed_centered,
                    moving_energy + moving_centered * moving_centered,
                )
            },
        );
    let denominator = fixed_energy.sqrt() * moving_energy.sqrt();
    if denominator == 0.0 {
        bail!("NCC is undefined for a constant volume");
    }
    Ok(numerator / denominator)
}

fn main() -> Result<()> {
    let ct_path = "D:/kwavers/leoneuro/data/brain_ct.nii.gz";
    let mr_path = "D:/kwavers/leoneuro/data/brain_mri_t1.nii.gz";
    let elastix_path = "D:/kwavers/leoneuro/scripts/elastix_result_mr_on_ct.nii.gz";
    let identity_output = "D:/kwavers/leoneuro/scripts/ritk_identity_mr_on_ct.nii.gz";
    let output = "D:/kwavers/leoneuro/scripts/registration_compare.png";

    let ct = NiftiReader::new(Backend::default()).read(ct_path)?;
    let mri = NiftiReader::new(Backend::default()).read(mr_path)?;
    let shape = ct.shape();
    let [nz, ny, nx] = shape;

    // The existing classical optimizer operates on index-space Leto volumes.
    let fixed_volume = image_to_leto_volume(&ct)?;
    let moving_volume = image_to_leto_volume(&mri)?;
    let registration = ImageRegistration::with_config(
        ClassicalConfig::default(),
        MutualInformationMetric::default(),
    );
    let result = registration.rigid_registration_mutual_info(
        &moving_volume,
        &fixed_volume,
        &AffineTransform::IDENTITY,
    )?;
    let trace = result.transform.as_array()[0]
        + result.transform.as_array()[5]
        + result.transform.as_array()[10];
    let angle_degrees = (((trace - 1.0) / 2.0).clamp(-1.0, 1.0)).acos().to_degrees();
    println!("RITK MI recovered rotation {angle_degrees:.2}°");

    // Convert the index-space result to physical space before native sampling.
    let physical_transform = index_affine_to_physical(&result.transform, &ct, &mri)?;
    let fixed_world = fixed_world_points(&ct);
    let identity = AtlasAffineTransform::<Backend, 3>::identity(None);
    let mr_identity = resample_moving_at_world(&fixed_world, &mri, &identity)?;
    let mr_ritk = resample_moving_at_world(&fixed_world, &mri, &physical_transform)?;
    let mr_elastix = host(&NiftiReader::new(Backend::default()).read(elastix_path)?)?;

    let identity_image = Image::from_flat_on(
        mr_identity.clone(),
        shape,
        *ct.origin(),
        *ct.spacing(),
        *ct.direction(),
        &Backend::default(),
    )?;
    ImageWriter::write(
        &NiftiWriter::new(Backend::default()),
        identity_output,
        &identity_image,
    )?;

    let ct_host = host(&ct)?;
    println!(
        "whole-volume NCC(CT, MR): identity {:.4}   RITK MI {:.4}   Elastix {:.4}",
        ncc(&ct_host, &mr_identity)?,
        ncc(&ct_host, &mr_ritk)?,
        ncc(&ct_host, &mr_elastix)?,
    );

    let axial_index = nz / 2;
    let ct_slice = slice_norm(&ct_host, shape, axial_index)?;
    let panels = [
        ("identity", slice_norm(&mr_identity, shape, axial_index)?),
        ("ritk-mi", slice_norm(&mr_ritk, shape, axial_index)?),
        ("elastix", slice_norm(&mr_elastix, shape, axial_index)?),
    ];

    let gap = 8_u32;
    let width = u32::try_from(nx).context("image width exceeds u32")?;
    let height = u32::try_from(ny).context("image height exceeds u32")?;
    let mut figure = RgbImage::from_pixel(width * 3 + gap * 2, height, Rgb([16, 16, 16]));
    for (panel_index, (_, moving_slice)) in panels.iter().enumerate() {
        let x_offset =
            u32::try_from(panel_index).context("panel index exceeds u32")? * (width + gap);
        for y in 0..ny {
            let output_y = u32::try_from(y).context("row index exceeds u32")?;
            for x in 0..nx {
                let output_x = x_offset + u32::try_from(x).context("column index exceeds u32")?;
                let index = y * nx + x;
                figure.put_pixel(
                    output_x,
                    output_y,
                    Rgb([
                        normalized_to_u8(ct_slice[index]),
                        normalized_to_u8(moving_slice[index]),
                        0,
                    ]),
                );
            }
        }
    }
    figure.save(output)?;
    println!(
        "wrote {output} (panels: identity | RITK MI | Elastix; R=CT, G=MR; axial z={axial_index})"
    );
    Ok(())
}
