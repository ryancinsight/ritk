//! Apollo-backed finite-boundary normalized cross-correlation.
//!
//! This path computes a linear correlation, not a circular one. The logical
//! moving-image ROI is explicitly zero-padded for the FFT, while candidate
//! means and energies come only from valid source samples. Candidates whose
//! complete block leaves the finite image or touches an unavailable validity
//! entry remains `-∞`, matching the direct metric's boundary contract.

use anyhow::{bail, Result};
use eunomia::Complex64;

use super::{
    BlockDisplacement, BlockMatchingConfig, MetricImage, MovingSamples, Sample, SubpixelRefinement,
};

/// Boundary padding policy for the FFT correlation path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftPadding {
    /// Use zero outside the finite source ROI during linear correlation.
    Zero,
}

/// Evaluate finite-boundary NCC with Apollo's FFT kernels.
///
/// This is the FFT equivalent of [`super::metric_image`]. It has the same
/// `[z, y, x]` row-major geometry and candidate-grid layout. The convolution
/// dimensions are padded to powers of two; no circular aliasing is included in
/// the returned search region.
///
/// # Errors
///
/// Returns an error for invalid geometry, mismatched buffers, arithmetic
/// overflow in the padded work shape, or a non-finite or featureless fixed
/// block. Moving candidates touching unavailable [`MovingSamples`] entries are
/// left at negative infinity.
pub fn metric_image_fft<T: Sample>(
    fixed: &[T],
    moving: MovingSamples<'_, T>,
    dims: [usize; 3],
    centre: [usize; 3],
    config: BlockMatchingConfig,
    padding: FftPadding,
) -> Result<MetricImage> {
    metric_image_fft_at(fixed, moving, dims, centre, centre, config, padding)
}

/// Match one block with the FFT metric and return its displacement.
///
/// This is a direct FFT counterpart to [`super::match_block`]. A propagated
/// moving centre should use the coarse-to-fine API once FFT selection is wired
/// into that policy layer.
pub fn match_block_fft<T: Sample>(
    fixed: &[T],
    moving: MovingSamples<'_, T>,
    dims: [usize; 3],
    centre: [usize; 3],
    config: BlockMatchingConfig,
    refinement: SubpixelRefinement,
    padding: FftPadding,
) -> Result<BlockDisplacement> {
    match_block_fft_at(
        fixed, moving, dims, centre, centre, config, refinement, padding,
    )
}

/// Match a block around separate fixed and moving centres with the FFT metric.
///
/// This crate-visible counterpart to [`match_block_fft`] is the execution seam
/// used by coarse-to-fine search. The returned displacement includes the
/// absolute moving-centre offset, just like `match_block_at` in the direct
/// metric path.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors match_block_at exactly; grouping the parameters here would               make the two seams diverge in shape without removing any of them"
)]
pub(crate) fn match_block_fft_at<T: Sample>(
    fixed: &[T],
    moving: MovingSamples<'_, T>,
    dims: [usize; 3],
    fixed_centre: [usize; 3],
    moving_centre: [usize; 3],
    config: BlockMatchingConfig,
    refinement: SubpixelRefinement,
    padding: FftPadding,
) -> Result<BlockDisplacement> {
    let surface = metric_image_fft_at(
        fixed,
        moving,
        dims,
        fixed_centre,
        moving_centre,
        config,
        padding,
    )?;
    let mut result = super::refine::displacement_from(&surface, refinement);
    for axis in 0..3 {
        result.displacement[axis] += moving_centre[axis] as f64 - fixed_centre[axis] as f64;
    }
    Ok(result)
}

/// Evaluate FFT NCC around `moving_centre` for a block fixed at
/// `fixed_centre`.
pub(crate) fn metric_image_fft_at<T: Sample>(
    fixed: &[T],
    moving: MovingSamples<'_, T>,
    dims: [usize; 3],
    fixed_centre: [usize; 3],
    moving_centre: [usize; 3],
    config: BlockMatchingConfig,
    padding: FftPadding,
) -> Result<MetricImage> {
    validate_inputs(fixed, moving, dims, fixed_centre, moving_centre, config)?;
    match padding {
        FftPadding::Zero => {}
    }

    let block_dims = checked_extents(config.block_radius, "block")?;
    let search_dims = checked_extents(config.search_radius, "search")?;
    let reach = [
        config.block_radius[0]
            .checked_add(config.search_radius[0])
            .ok_or_else(|| anyhow::anyhow!("block/search radius overflows on axis 0"))?,
        config.block_radius[1]
            .checked_add(config.search_radius[1])
            .ok_or_else(|| anyhow::anyhow!("block/search radius overflows on axis 1"))?,
        config.block_radius[2]
            .checked_add(config.search_radius[2])
            .ok_or_else(|| anyhow::anyhow!("block/search radius overflows on axis 2"))?,
    ];
    let roi_dims = checked_extents(reach, "moving ROI")?;
    let convolution_dims = [
        roi_dims[0]
            .checked_add(block_dims[0] - 1)
            .ok_or_else(|| anyhow::anyhow!("FFT convolution extent overflows on axis 0"))?,
        roi_dims[1]
            .checked_add(block_dims[1] - 1)
            .ok_or_else(|| anyhow::anyhow!("FFT convolution extent overflows on axis 1"))?,
        roi_dims[2]
            .checked_add(block_dims[2] - 1)
            .ok_or_else(|| anyhow::anyhow!("FFT convolution extent overflows on axis 2"))?,
    ];
    let fft_dims = [
        next_power_of_two(convolution_dims[0], "axis 0")?,
        next_power_of_two(convolution_dims[1], "axis 1")?,
        next_power_of_two(convolution_dims[2], "axis 2")?,
    ];
    let fft_len = checked_product(fft_dims, "FFT buffer")?;

    let fixed_values = gather_fixed_block(fixed, dims, fixed_centre, config.block_radius);
    if fixed_values.iter().any(|value| !value.is_finite()) {
        bail!(
            "fixed block at {fixed_centre:?} contains a non-finite sample; every candidate would depend on unavailable data"
        );
    }
    let fixed_mean = fixed_values.iter().sum::<f64>() / fixed_values.len() as f64;
    let fixed_energy = fixed_values
        .iter()
        .map(|&value| {
            let centred = value - fixed_mean;
            centred * centred
        })
        .sum::<f64>();
    if fixed_energy <= 0.0 {
        bail!(
            "fixed block at {fixed_centre:?} has zero variance; normalized correlation is undefined"
        );
    }

    let mut moving_spectrum = vec![Complex64::new(0.0, 0.0); fft_len];
    let mut fixed_spectrum = vec![Complex64::new(0.0, 0.0); fft_len];
    let roi_origin = [
        isize::try_from(moving_centre[0])? - isize::try_from(reach[0])?,
        isize::try_from(moving_centre[1])? - isize::try_from(reach[1])?,
        isize::try_from(moving_centre[2])? - isize::try_from(reach[2])?,
    ];

    // The logical ROI and the remaining FFT buffer are initialized to zero.
    // Source samples are copied only where the logical ROI overlaps the image
    // and the caller marks them available. Invalid finite sentinels must not
    // enter the global FFT because they can contaminate correlations for
    // otherwise disjoint valid candidates through finite-precision roundoff.
    let moving_validity = moving.validity();
    for z in 0..roi_dims[0] {
        let source_z = roi_origin[0] + z as isize;
        for y in 0..roi_dims[1] {
            let source_y = roi_origin[1] + y as isize;
            for x in 0..roi_dims[2] {
                let source_x = roi_origin[2] + x as isize;
                if source_z < 0
                    || source_y < 0
                    || source_x < 0
                    || source_z >= dims[0] as isize
                    || source_y >= dims[1] as isize
                    || source_x >= dims[2] as isize
                {
                    continue;
                }
                let source =
                    (source_z as usize * dims[1] + source_y as usize) * dims[2] + source_x as usize;
                let sample = moving.values()[source].to_f64();
                if moving_validity.is_none_or(|validity| validity[source]) && sample.is_finite() {
                    moving_spectrum[flat_index([z, y, x], fft_dims)] = Complex64::new(sample, 0.0);
                }
            }
        }
    }

    // Reverse the fixed block so convolution samples the moving block at each
    // candidate start position.
    for z in 0..block_dims[0] {
        for y in 0..block_dims[1] {
            for x in 0..block_dims[2] {
                let source = flat_index([z, y, x], block_dims);
                let reversed = [
                    block_dims[0] - 1 - z,
                    block_dims[1] - 1 - y,
                    block_dims[2] - 1 - x,
                ];
                fixed_spectrum[flat_index(reversed, fft_dims)] =
                    Complex64::new(fixed_values[source], 0.0);
            }
        }
    }

    fft3d(&mut moving_spectrum, fft_dims, false);
    fft3d(&mut fixed_spectrum, fft_dims, false);
    for (moving_bin, fixed_bin) in moving_spectrum.iter_mut().zip(&fixed_spectrum) {
        *moving_bin = Complex64::new(
            moving_bin.re * fixed_bin.re - moving_bin.im * fixed_bin.im,
            moving_bin.re * fixed_bin.im + moving_bin.im * fixed_bin.re,
        );
    }
    fft3d(&mut moving_spectrum, fft_dims, true);
    let inverse_scale = 1.0 / fft_len as f64;

    let mut values = vec![f64::NEG_INFINITY; checked_product(search_dims, "metric image")?];
    let mut candidate = Vec::with_capacity(fixed_values.len());
    for (oz, dz) in
        (-(config.search_radius[0] as isize)..=config.search_radius[0] as isize).enumerate()
    {
        for (oy, dy) in
            (-(config.search_radius[1] as isize)..=config.search_radius[1] as isize).enumerate()
        {
            for (ox, dx) in
                (-(config.search_radius[2] as isize)..=config.search_radius[2] as isize).enumerate()
            {
                let offset = [dz, dy, dx];
                if !candidate_inside(moving_centre, offset, dims, config.block_radius) {
                    continue;
                }

                let candidate_centre = [
                    (moving_centre[0] as isize + dz) as usize,
                    (moving_centre[1] as isize + dy) as usize,
                    (moving_centre[2] as isize + dx) as usize,
                ];
                if !super::metric::candidate_is_valid(
                    moving.validity(),
                    dims,
                    candidate_centre,
                    config.block_radius,
                ) {
                    continue;
                }

                gather_candidate(
                    moving.values(),
                    dims,
                    moving_centre,
                    offset,
                    config.block_radius,
                    &mut candidate,
                );
                if candidate.iter().any(|value| !value.is_finite()) {
                    continue;
                }
                let sum = candidate.iter().sum::<f64>();
                let sum_squared = candidate.iter().map(|&value| value * value).sum::<f64>();
                let variance_sum = sum_squared - sum * sum / fixed_values.len() as f64;
                if variance_sum <= 0.0 {
                    continue;
                }

                let convolution_index = [
                    (config.search_radius[0] as isize + dz) as usize + block_dims[0] - 1,
                    (config.search_radius[1] as isize + dy) as usize + block_dims[1] - 1,
                    (config.search_radius[2] as isize + dx) as usize + block_dims[2] - 1,
                ];
                let raw_cross =
                    moving_spectrum[flat_index(convolution_index, fft_dims)].re * inverse_scale;
                let centred_cross = raw_cross - fixed_mean * sum;
                values[(oz * search_dims[1] + oy) * search_dims[2] + ox] =
                    centred_cross / (fixed_energy * variance_sum).sqrt();
            }
        }
    }

    Ok(MetricImage {
        values,
        extent: search_dims,
        search_radius: config.search_radius,
    })
}

fn validate_inputs<T: Sample>(
    fixed: &[T],
    moving: MovingSamples<'_, T>,
    dims: [usize; 3],
    fixed_centre: [usize; 3],
    moving_centre: [usize; 3],
    config: BlockMatchingConfig,
) -> Result<()> {
    config.validate()?;
    let expected = checked_product(dims, "image")?;
    if fixed.len() != expected || moving.values().len() != expected {
        bail!(
            "fixed ({}) and moving ({}) buffers must both hold {expected} voxels for dims {dims:?}",
            fixed.len(),
            moving.values().len()
        );
    }
    for &dimension in &dims {
        isize::try_from(dimension)?;
    }
    for axis in 0..3 {
        let fixed_hi = fixed_centre[axis]
            .checked_add(config.block_radius[axis])
            .ok_or_else(|| anyhow::anyhow!("fixed centre overflows on axis {axis}"))?;
        if fixed_centre[axis]
            .checked_sub(config.block_radius[axis])
            .is_none()
            || fixed_hi >= dims[axis]
        {
            bail!("fixed block at {fixed_centre:?} leaves the image on axis {axis}");
        }
        let moving_hi = moving_centre[axis]
            .checked_add(config.block_radius[axis])
            .ok_or_else(|| anyhow::anyhow!("moving centre overflows on axis {axis}"))?;
        if moving_centre[axis]
            .checked_sub(config.block_radius[axis])
            .is_none()
            || moving_hi >= dims[axis]
        {
            bail!("moving block at {moving_centre:?} leaves the image on axis {axis}");
        }
        isize::try_from(config.block_radius[axis])?;
        isize::try_from(config.search_radius[axis])?;
    }
    Ok(())
}

fn checked_product(dims: [usize; 3], label: &str) -> Result<usize> {
    dims[0]
        .checked_mul(dims[1])
        .and_then(|value| value.checked_mul(dims[2]))
        .ok_or_else(|| anyhow::anyhow!("{label} dimensions {dims:?} overflow"))
}

fn checked_extents(radius: [usize; 3], label: &str) -> Result<[usize; 3]> {
    Ok([
        radius[0]
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("{label} extent overflows on axis 0"))?,
        radius[1]
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("{label} extent overflows on axis 1"))?,
        radius[2]
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("{label} extent overflows on axis 2"))?,
    ])
}

fn next_power_of_two(value: usize, axis: &str) -> Result<usize> {
    value
        .checked_next_power_of_two()
        .ok_or_else(|| anyhow::anyhow!("FFT padded extent overflows on {axis}"))
}

fn flat_index([z, y, x]: [usize; 3], dims: [usize; 3]) -> usize {
    (z * dims[1] + y) * dims[2] + x
}

fn gather_fixed_block<T: Sample>(
    fixed: &[T],
    dims: [usize; 3],
    centre: [usize; 3],
    radius: [usize; 3],
) -> Vec<f64> {
    let mut values =
        Vec::with_capacity((2 * radius[0] + 1) * (2 * radius[1] + 1) * (2 * radius[2] + 1));
    for z in centre[0] - radius[0]..=centre[0] + radius[0] {
        for y in centre[1] - radius[1]..=centre[1] + radius[1] {
            for x in centre[2] - radius[2]..=centre[2] + radius[2] {
                values.push(fixed[(z * dims[1] + y) * dims[2] + x].to_f64());
            }
        }
    }
    values
}

fn candidate_inside(
    centre: [usize; 3],
    offset: [isize; 3],
    dims: [usize; 3],
    radius: [usize; 3],
) -> bool {
    (0..3).all(|axis| {
        let centre = centre[axis] as isize + offset[axis];
        let radius = radius[axis] as isize;
        centre - radius >= 0 && centre + radius < dims[axis] as isize
    })
}

fn gather_candidate<T: Sample>(
    moving: &[T],
    dims: [usize; 3],
    centre: [usize; 3],
    offset: [isize; 3],
    radius: [usize; 3],
    out: &mut Vec<f64>,
) {
    out.clear();
    let shifted = [
        (centre[0] as isize + offset[0]) as usize,
        (centre[1] as isize + offset[1]) as usize,
        (centre[2] as isize + offset[2]) as usize,
    ];
    for z in shifted[0] - radius[0]..=shifted[0] + radius[0] {
        for y in shifted[1] - radius[1]..=shifted[1] + radius[1] {
            for x in shifted[2] - radius[2]..=shifted[2] + radius[2] {
                out.push(moving[(z * dims[1] + y) * dims[2] + x].to_f64());
            }
        }
    }
}

fn fft3d(data: &mut [Complex64], dims: [usize; 3], inverse: bool) {
    let plane = dims[1] * dims[2];
    for z in 0..dims[0] {
        for y in 0..dims[1] {
            transform_line(
                &mut data[z * plane + y * dims[2]..z * plane + (y + 1) * dims[2]],
                inverse,
            );
        }
    }

    let mut line = vec![Complex64::new(0.0, 0.0); dims[1]];
    for z in 0..dims[0] {
        for x in 0..dims[2] {
            for y in 0..dims[1] {
                line[y] = data[z * plane + y * dims[2] + x];
            }
            transform_line(&mut line, inverse);
            for y in 0..dims[1] {
                data[z * plane + y * dims[2] + x] = line[y];
            }
        }
    }

    let mut line = vec![Complex64::new(0.0, 0.0); dims[0]];
    for y in 0..dims[1] {
        for x in 0..dims[2] {
            for z in 0..dims[0] {
                line[z] = data[z * plane + y * dims[2] + x];
            }
            transform_line(&mut line, inverse);
            for z in 0..dims[0] {
                data[z * plane + y * dims[2] + x] = line[z];
            }
        }
    }
}

fn transform_line(line: &mut [Complex64], inverse: bool) {
    if line.len() <= 1 {
        return;
    }
    if inverse {
        apollo_fft::application::execution::kernel::fft_inverse_unnorm(line);
    } else {
        apollo_fft::application::execution::kernel::fft_forward(line);
    }
}
