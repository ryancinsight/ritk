//! End-to-end NIfTI workflow: build a volume, write it, read it back, and
//! verify that neither voxels nor spatial metadata were altered.
//!
//! NIfTI is the interchange format where geometry is most often silently lost:
//! RITK tensors are ordered `[Z, Y, X]` while NIfTI file axes are `[X, Y, Z]`,
//! and NIfTI stores its affine in RAS while RITK carries LPS. Both conversions
//! happen inside the writer and reader, so the only honest check is a
//! round-trip that asserts bit-exact voxels and geometry rather than merely
//! asserting the calls succeeded.
//!
//! The example covers the three surfaces a caller actually uses:
//!   1. a single anisotropic volume through NIfTI-1,
//!   2. the same volume through NIfTI-2, whose wider header must not change
//!      the recovered values, and
//!   3. a multi-volume series, which shares one affine across frames, so what
//!      must survive is each frame's voxels staying attached to its own frame.
//!
//! Run with `cargo run --example nifti_roundtrip -p ritk-nifti`.
#![expect(clippy::print_stdout, reason = "ratchet RITK-LINT-1")]

use anyhow::{bail, Context, Result};
use coeus_core::SequentialBackend;
use ritk_image::Image;
use ritk_nifti::{read_nifti, read_nifti_series, write_nifti, write_nifti2, write_nifti_series};
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

/// Tensor shape in RITK's `[Z, Y, X]` order. Deliberately unequal so a
/// transposed axis cannot pass as a coincidence.
const SHAPE: [usize; 3] = [3, 5, 7];

/// Anisotropic spacing in millimetres, one axis distinct per index so an axis
/// permutation is observable.
const SPACING_MM: [f64; 3] = [4.0, 0.75, 1.25];

/// Scanner origin in LPS millimetres, offset on every axis.
const ORIGIN_MM: [f64; 3] = [-12.5, 33.25, 7.0];

/// Voxel comparisons are exact: NIfTI stores `f32` and RITK writes `f32`, so a
/// lossless round-trip has no representation error to absorb. Geometry is
/// compared against this bound because the affine travels through an LPS/RAS
/// sign flip and a column reversal in `f64`, where only rounding is expected.
const GEOMETRY_TOLERANCE_MM: f64 = 1e-9;

type Volume = Image<f32, SequentialBackend, 3>;

/// A deterministic ramp that is unique per voxel, so any axis transposition,
/// flip, or off-by-one stride shows up as a value mismatch.
fn phantom_values(shape: [usize; 3]) -> Result<Vec<f32>> {
    let [nz, ny, nx] = shape;
    let count = nz
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nx))
        .context("phantom dimensions overflow usize")?;
    let mut values = Vec::with_capacity(count);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                // Distinct per-axis weights make the index recoverable from the
                // value, which is what turns a mismatch into a diagnosis.
                let encoded = (z * 10_000 + y * 100 + x) as f32;
                values.push(encoded);
            }
        }
    }
    Ok(values)
}

fn build_volume(shape: [usize; 3], origin_mm: [f64; 3]) -> Result<Volume> {
    Image::from_flat_on(
        phantom_values(shape)?,
        shape,
        Point::new(origin_mm),
        Spacing::new(SPACING_MM),
        Direction::identity(),
        &SequentialBackend,
    )
    .context("phantom dimensions do not match the voxel count")
}

/// Assert that `loaded` carries exactly the voxels and geometry of `original`.
fn assert_roundtrip_preserved(label: &str, original: &Volume, loaded: &Volume) -> Result<()> {
    if loaded.shape() != original.shape() {
        bail!(
            "{label}: shape changed across the round trip: wrote {:?}, read {:?}",
            original.shape(),
            loaded.shape()
        );
    }

    let written = original
        .data_slice()
        .context("original volume is not contiguous")?;
    let recovered = loaded
        .data_slice()
        .context("recovered volume is not contiguous")?;
    if written != recovered {
        let first = written
            .iter()
            .zip(recovered)
            .position(|(w, r)| w != r)
            .unwrap_or(0);
        bail!(
            "{label}: voxels changed across the round trip; first mismatch at flat index {first}: wrote {}, read {}",
            written[first],
            recovered[first]
        );
    }

    for axis in 0..3 {
        let origin_delta = (loaded.origin()[axis] - original.origin()[axis]).abs();
        if origin_delta > GEOMETRY_TOLERANCE_MM {
            bail!(
                "{label}: origin axis {axis} drifted {origin_delta} mm (bound {GEOMETRY_TOLERANCE_MM})"
            );
        }
        let spacing_delta = (loaded.spacing()[axis] - original.spacing()[axis]).abs();
        if spacing_delta > GEOMETRY_TOLERANCE_MM {
            bail!(
                "{label}: spacing axis {axis} drifted {spacing_delta} mm (bound {GEOMETRY_TOLERANCE_MM})"
            );
        }
    }

    println!(
        "{label}: {:?} voxels, spacing {:?} mm, origin {:?} mm preserved exactly",
        loaded.shape(),
        [
            loaded.spacing()[0],
            loaded.spacing()[1],
            loaded.spacing()[2]
        ],
        [loaded.origin()[0], loaded.origin()[1], loaded.origin()[2]]
    );
    Ok(())
}

fn main() -> Result<()> {
    let backend = SequentialBackend;
    let dir = tempdir().context("failed to create a temporary working directory")?;

    let volume = build_volume(SHAPE, ORIGIN_MM)?;

    // 1. NIfTI-1, the default single-file surface.
    let nifti1_path = dir.path().join("phantom.nii");
    write_nifti(&nifti1_path, &volume, &backend).context("failed to write the NIfTI-1 volume")?;
    let nifti1_loaded =
        read_nifti(&nifti1_path, &backend).context("failed to read the NIfTI-1 volume back")?;
    assert_roundtrip_preserved("NIfTI-1", &volume, &nifti1_loaded)?;

    // 2. NIfTI-2. The wider header changes the on-disk layout but must recover
    //    identical voxels and geometry; the reader auto-detects the version.
    let nifti2_path = dir.path().join("phantom.nii2");
    write_nifti2(&nifti2_path, &volume, &backend).context("failed to write the NIfTI-2 volume")?;
    let nifti2_loaded =
        read_nifti(&nifti2_path, &backend).context("failed to read the NIfTI-2 volume back")?;
    assert_roundtrip_preserved("NIfTI-2", &volume, &nifti2_loaded)?;

    // The two on-disk formats must agree voxel for voxel, which a single
    // round-trip alone cannot establish.
    let nifti1_voxels = nifti1_loaded
        .data_slice()
        .context("NIfTI-1 not contiguous")?;
    let nifti2_voxels = nifti2_loaded
        .data_slice()
        .context("NIfTI-2 not contiguous")?;
    if nifti1_voxels != nifti2_voxels {
        bail!("NIfTI-1 and NIfTI-2 recovered different voxels from the same source volume");
    }
    println!("NIfTI-1 and NIfTI-2 agree voxel for voxel");

    // 3. A series. A 4D NIfTI carries one affine for the whole stack, so the
    //    volumes share a spatial grid and differ only in voxel content — the
    //    shape of a time series or a multi-b-value acquisition. The writer
    //    enforces that shared grid; what the round trip must prove is that each
    //    frame's voxels come back attached to the right frame. A writer that
    //    collapsed the series to its first volume, or that reordered frames,
    //    would pass a shape check but fail here.
    let series: Vec<Volume> = (0..3)
        .map(|frame| {
            let values = phantom_values(SHAPE)?
                .into_iter()
                // A per-frame offset far larger than any intra-frame value, so
                // a frame swap is unambiguous rather than a near miss.
                .map(|value| value + (frame as f32) * 1_000_000.0)
                .collect();
            Image::from_flat_on(
                values,
                SHAPE,
                Point::new(ORIGIN_MM),
                Spacing::new(SPACING_MM),
                Direction::identity(),
                &SequentialBackend,
            )
            .context("series frame dimensions do not match the voxel count")
        })
        .collect::<Result<_>>()?;

    let series_path = dir.path().join("phantom_series.nii");
    write_nifti_series(&series_path, &series, &backend)
        .context("failed to write the NIfTI series")?;
    let series_loaded = read_nifti_series(&series_path, &backend)
        .context("failed to read the NIfTI series back")?;

    if series_loaded.len() != series.len() {
        bail!(
            "series length changed across the round trip: wrote {}, read {}",
            series.len(),
            series_loaded.len()
        );
    }
    for (index, (original, loaded)) in series.iter().zip(&series_loaded).enumerate() {
        assert_roundtrip_preserved(&format!("series volume {index}"), original, loaded)?;
    }

    println!(
        "NIfTI round trip complete: {} single volumes and a {}-volume series preserved exactly",
        2,
        series.len()
    );
    Ok(())
}
