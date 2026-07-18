//! Label-preserving Euclidean morphology, matching ITK's `LabelErodeDilate`
//! module (`itk::LabelSetDilateImageFilter` / `itk::LabelSetErodeImageFilter`).
//!
//! # Mathematical Specification
//!
//! Each distinct non-zero value in the input is a label region.  Dilation grows
//! every region by a Euclidean structuring element of per-axis radius `r`,
//! resolving overlaps in favor of the nearer region (the parabola contact point);
//! erosion shrinks every region by the same element, keeping only voxels whose
//! Euclidean distance to their region boundary is â‰¥ `r`.  Unlike independent
//! per-label binary morphology, labels never bleed into one another: the result
//! is a single label image.
//!
//! The implementation is the separable parabolic (lower/upper-envelope) algorithm
//! of Beare: a squared-distance image is propagated one axis at a time, carrying
//! the winning label alongside.  `m_Scale[d] = 0.5Â·r_dÂ²` (with image spacing) or
//! `0.5Â·r_dÂ² + 1` (voxel units â€” the `+1` includes voxels exactly `r` away).
//! All real arithmetic is `f64` (`NumericTraits<int>::FloatType == double`),
//! matching ITK bit-for-bit.
//!
//! # Axis convention
//! ITK iterates dimensions `x, y, z`; the ritk tensor is `[z, y, x]`, so ITK
//! dimension `d` is ritk axis `2 âˆ’ d`.  Radius and spacing are taken in ITK order
//! `[x, y, z]`.  A `z = 1` volume (2-D image) yields length-1 z-lines that are
//! no-ops, so the 3-D path reproduces SimpleITK's 2-D result exactly.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Whether to grow (`Dilate`) or shrink (`Erode`) the label regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelSetMorphOp {
    /// Grow each label region by the Euclidean structuring element.
    Dilate,
    /// Shrink each label region by the Euclidean structuring element.
    Erode,
}

/// First-pass dilation along a line: `dist`/`lab` are gathered line buffers
/// (label values), producing the propagated distance and label.  Ports
/// `DoLineDilateFirstPass`.
fn dilate_first_line(
    lab_in: &[f32],
    sigma: f64,
    magnitude: f64,
    dist: &mut [f64],
    lab_out: &mut [f32],
    line: &mut [f64],
    tmp: &mut [f64],
) {
    let n = lab_in.len();
    // line buffer initialized: sigma at labels, 0 at background.
    for (l, &lab) in line.iter_mut().zip(lab_in) {
        *l = if lab != 0.0 { sigma } else { 0.0 };
    }

    // left pass â†’ writes `tmp` and `lab_out`.
    let mut lastcontact: usize = 0;
    let mut lastval = line[0];
    for pos in 0..n {
        let krange = pos as f64 - lastcontact as f64;
        let thisval = lastval - magnitude * krange * krange;
        if line[pos] >= line[lastcontact] {
            lastcontact = pos;
            lastval = line[pos];
        }
        tmp[pos] = line[pos].max(thisval);
        lab_out[pos] = if thisval > line[pos] {
            lab_in[lastcontact]
        } else {
            lab_in[pos]
        };
    }
    // right pass â†’ reads `tmp`, writes `dist` and conditionally `lab_out`.
    lastcontact = n - 1;
    lastval = tmp[n - 1];
    for pos in (0..n).rev() {
        let krange = lastcontact as f64 - pos as f64;
        let thisval = lastval - magnitude * krange * krange;
        if tmp[pos] >= tmp[lastcontact] {
            lastcontact = pos;
            lastval = tmp[pos];
        }
        dist[pos] = tmp[pos].max(thisval);
        if thisval > tmp[pos] {
            lab_out[pos] = lab_in[lastcontact];
        }
    }
}

/// Subsequent-pass dilation along a line (contact-point label propagation).
/// Ports `DoLineLabelProp<â€¦, doDilate=true>`.  `dist`/`lab` are updated in place.
fn dilate_prop_line(
    dist: &mut [f64],
    lab: &mut [f32],
    magnitude: f64,
    extreme: f64,
    tmp: &mut [f64],
    tmp_lab: &mut [f32],
) {
    let n = dist.len();

    // negative half
    let mut koffset: isize = 0;
    let mut newcontact: isize = 0;
    for pos in 0..n {
        let mut base_val = extreme;
        let mut base_lab = lab[pos];
        let mut krange = koffset;
        while krange <= 0 {
            let idx = pos as isize + krange;
            let t = dist[idx as usize] - magnitude * (krange * krange) as f64;
            if t >= base_val {
                base_val = t;
                newcontact = krange;
                base_lab = lab[idx as usize];
            }
            krange += 1;
        }
        tmp[pos] = base_val;
        tmp_lab[pos] = base_lab;
        koffset = newcontact - 1;
    }
    // positive half
    koffset = 0;
    newcontact = 0;
    for pos in (0..n).rev() {
        let mut base_val = extreme;
        let mut base_lab = tmp_lab[pos];
        let mut krange = koffset;
        while krange >= 0 {
            let idx = pos as isize + krange;
            let t = tmp[idx as usize] - magnitude * (krange * krange) as f64;
            if t >= base_val {
                base_val = t;
                newcontact = krange;
                base_lab = tmp_lab[idx as usize];
            }
            krange -= 1;
        }
        dist[pos] = base_val;
        lab[pos] = base_lab;
        koffset = newcontact + 1;
    }
}

/// First-pass erosion of a single label run of length `sll`. Ports
/// `DoLineErodeFirstPass` (writes squared distance into `seg`).
fn erode_first_run(
    sll: usize,
    leftend: f64,
    rightend: f64,
    magnitude: f64,
    sigma: f64,
    seg: &mut [f64],
) {
    for (pos, s) in seg.iter_mut().enumerate() {
        let offset = (sll - pos) as f64;
        let left = leftend - magnitude * (pos as f64 + 1.0) * (pos as f64 + 1.0);
        let right = rightend - magnitude * offset * offset;
        *s = left.min(right).min(sigma);
    }
}

/// Subsequent-pass erosion of a padded run (`buf` length `sll+2`, sentinels at
/// the ends). Ports `DoLine<â€¦, doDilate=false>`.
fn erode_line_run(buf: &mut [f64], magnitude: f64, extreme: f64) {
    let n = buf.len();
    let mut tmp = vec![0.0_f64; n];
    // negative half
    let mut koffset: isize = 0;
    let mut newcontact: isize = 0;
    // index-based: the inner loop reads `buf`/`tmp` at the computed contact
    // offset `pos + krange`, so a positional iterator does not apply.
    #[allow(clippy::needless_range_loop)]
    for pos in 0..n {
        let mut base_val = extreme;
        let mut krange = koffset;
        while krange <= 0 {
            let idx = pos as isize + krange;
            let t = buf[idx as usize] - magnitude * (krange * krange) as f64;
            if t <= base_val {
                base_val = t;
                newcontact = krange;
            }
            krange += 1;
        }
        tmp[pos] = base_val;
        koffset = newcontact - 1;
    }
    // positive half
    koffset = 0;
    newcontact = 0;
    for pos in (0..n).rev() {
        let mut base_val = extreme;
        let mut krange = koffset;
        while krange >= 0 {
            let idx = pos as isize + krange;
            let t = tmp[idx as usize] - magnitude * (krange * krange) as f64;
            if t <= base_val {
                base_val = t;
                newcontact = krange;
            }
            krange -= 1;
        }
        buf[pos] = base_val;
        koffset = newcontact + 1;
    }
}

/// Run-length-encode the maximal runs of equal non-zero label in a line.
fn label_runs(lab: &[f32]) -> Vec<(usize, usize)> {
    let n = lab.len();
    let mut runs = Vec::new();
    let mut idx = 0;
    while idx < n {
        let v = lab[idx];
        if v != 0.0 {
            let mut end = idx;
            while end < n && lab[end] == v {
                end += 1;
            }
            runs.push((idx, end - 1));
            idx = end;
        } else {
            idx += 1;
        }
    }
    runs
}

/// Visit every line along ritk `axis` of a `[Z,Y,X]` buffer, gathering it into a
/// contiguous scratch vector, applying `f`, and scattering it back.
fn for_each_line<F: FnMut(&mut Vec<f32>, &mut Vec<f64>)>(
    dims: [usize; 3],
    axis: usize,
    lab: &mut [f32],
    dist: &mut [f64],
    mut f: F,
) {
    let strides = [dims[1] * dims[2], dims[2], 1usize];
    let len = dims[axis];
    let stride = strides[axis];
    // The two orthogonal axes enumerate line starts.
    let others: Vec<usize> = (0..3).filter(|&a| a != axis).collect();
    let (a0, a1) = (others[0], others[1]);
    let mut line_lab = vec![0.0_f32; len];
    let mut line_dist = vec![0.0_f64; len];
    for i0 in 0..dims[a0] {
        for i1 in 0..dims[a1] {
            let mut coord = [0usize; 3];
            coord[a0] = i0;
            coord[a1] = i1;
            let start = coord[0] * strides[0] + coord[1] * strides[1] + coord[2] * strides[2];
            for k in 0..len {
                let off = start + k * stride;
                line_lab[k] = lab[off];
                line_dist[k] = dist[off];
            }
            f(&mut line_lab, &mut line_dist);
            for k in 0..len {
                let off = start + k * stride;
                lab[off] = line_lab[k];
                dist[off] = line_dist[k];
            }
        }
    }
}

/// Apply label-preserving Euclidean dilation or erosion.
///
/// # Arguments
/// - `input`: integer label image (0 = background).
/// - `radius_itk`: per-axis radius in ITK order `[x, y, z]`.
/// - `use_spacing`: interpret radius in world units (`true`, ITK/SimpleITK
///   default) or voxels (`false`, adds a one-voxel inclusion margin).
///
/// # Postcondition
/// Output equals `sitk.LabelSetDilate`/`LabelSetErode` with the same arguments.
pub fn label_set_morph<B: Backend>(
    input: &Image<f32, B, 3>,
    radius_itk: [f64; 3],
    use_spacing: bool,
    op: LabelSetMorphOp,
) -> Image<f32, B, 3> {
    let (mut lab, dims) = extract_vec_infallible(input);
    let n = lab.len();
    let dims3 = [dims[0], dims[1], dims[2]];

    // image spacing is stored [z,y,x]; ITK order is [x,y,z].
    let sp = input.spacing();
    let spacing_itk = [sp[2], sp[1], sp[0]];

    let do_dilate = op == LabelSetMorphOp::Dilate;
    let sign = if do_dilate { 1.0 } else { -1.0 };
    let extreme = if do_dilate { f64::MIN } else { f64::MAX };

    // A z=1 volume is a 2-D image (ITK ImageDimension == 2): only x, y are
    // active.  A genuine 3-D volume processes x, y, z.
    let ndim = if dims3[0] == 1 { 2 } else { 3 };

    // m_Scale per ITK dimension (only the active dims matter).
    let mut scale = [0.0_f64; 3];
    for (p, s) in scale.iter_mut().enumerate().take(ndim) {
        let r = radius_itk[p];
        *s = if use_spacing {
            0.5 * r * r
        } else {
            0.5 * r * r + 1.0
        };
    }
    // firstval = first dim with non-zero radius.
    let firstval = match (0..ndim).find(|&p| radius_itk[p] != 0.0) {
        Some(f) => f,
        None => return input.clone(),
    };
    let base_sigma = scale[firstval];
    for s in scale.iter_mut().take(ndim).skip(firstval + 1) {
        *s /= base_sigma;
    }

    let mut dist = vec![0.0_f64; n];
    let mut first_pass_done = false;

    // Per-scanline scratch, allocated once and reused across every line of every
    // axis (sized to the longest axis) â€” the separable kernels write into slices
    // of these instead of allocating fresh buffers per line.
    let maxlen = *dims3.iter().max().unwrap_or(&0);
    let mut s_line = vec![0.0_f64; maxlen];
    let mut s_tmp = vec![0.0_f64; maxlen];
    let mut s_f32 = vec![0.0_f32; maxlen];

    for d in 0..ndim {
        if scale[d] <= 0.0 {
            continue;
        }
        let axis = 2 - d;
        let sigma = scale[d];
        let iscale = if use_spacing { spacing_itk[d] } else { 1.0 };
        let last = d == ndim - 1;

        if !first_pass_done {
            let magnitude = sign * iscale * iscale / 2.0;
            if do_dilate {
                for_each_line(dims3, axis, &mut lab, &mut dist, |line_lab, line_dist| {
                    let len = line_lab.len();
                    dilate_first_line(
                        line_lab,
                        sigma,
                        magnitude,
                        line_dist,
                        &mut s_f32[..len],
                        &mut s_line[..len],
                        &mut s_tmp[..len],
                    );
                    line_lab.copy_from_slice(&s_f32[..len]);
                });
            } else {
                for_each_line(dims3, axis, &mut lab, &mut dist, |line_lab, line_dist| {
                    let len = line_lab.len();
                    erode_first_dim(
                        line_lab,
                        line_dist,
                        sigma,
                        magnitude,
                        base_sigma,
                        last,
                        &mut s_line[..len],
                    );
                });
            }
        } else {
            let magnitude = sign * iscale * iscale / (2.0 * sigma);
            if do_dilate {
                for_each_line(dims3, axis, &mut lab, &mut dist, |line_lab, line_dist| {
                    let len = line_lab.len();
                    dilate_prop_line(
                        line_dist,
                        line_lab,
                        magnitude,
                        extreme,
                        &mut s_tmp[..len],
                        &mut s_f32[..len],
                    );
                });
            } else {
                for_each_line(dims3, axis, &mut lab, &mut dist, |line_lab, line_dist| {
                    let len = line_lab.len();
                    erode_subsequent_dim(
                        line_lab,
                        line_dist,
                        magnitude,
                        extreme,
                        base_sigma,
                        last,
                        &mut s_line[..len],
                    );
                });
            }
        }
        first_pass_done = true;
    }

    rebuild(lab, dims, input)
}

/// First erosion dimension: run-length encode, erode each run into the distance
/// buffer, optionally threshold labels (`last` pass). Ports
/// `doOneDimensionErodeFirstPass`.
#[allow(clippy::too_many_arguments)]
fn erode_first_dim(
    line_lab: &mut [f32],
    line_dist: &mut [f64],
    sigma: f64,
    magnitude: f64,
    base_sigma: f64,
    last: bool,
    line: &mut [f64],
) {
    let n = line_lab.len();
    // lineBuf init: 1.0 at labels (per ITK), but the per-run distances overwrite.
    for (l, &lab) in line.iter_mut().zip(line_lab.iter()) {
        *l = if lab != 0.0 { 1.0 } else { 0.0 };
    }
    for (first, lastpos) in label_runs(line_lab) {
        let sll = lastpos - first + 1;
        let leftend = if first == 0 { sigma } else { 0.0 };
        let rightend = if lastpos == n - 1 { sigma } else { 0.0 };
        let mut seg = vec![0.0_f64; sll];
        erode_first_run(sll, leftend, rightend, magnitude, sigma, &mut seg);
        line[first..=lastpos].copy_from_slice(&seg);
    }
    line_dist.copy_from_slice(line);
    if last {
        for i in 0..n {
            line_lab[i] = if line[i] == base_sigma {
                line_lab[i]
            } else {
                0.0
            };
        }
    }
}

/// Subsequent erosion dimension: run-length encode, erode padded runs, threshold
/// on the last pass. Ports `doOneDimensionErode`.
#[allow(clippy::too_many_arguments)]
fn erode_subsequent_dim(
    line_lab: &mut [f32],
    line_dist: &mut [f64],
    magnitude: f64,
    extreme: f64,
    base_sigma: f64,
    last: bool,
    line: &mut [f64],
) {
    let n = line_lab.len();
    line.copy_from_slice(line_dist);
    for (first, lastpos) in label_runs(line_lab) {
        let sll = lastpos - first + 1;
        let leftend = if first == 0 { base_sigma } else { 0.0 };
        let rightend = if lastpos == n - 1 { base_sigma } else { 0.0 };
        let mut buf = vec![0.0_f64; sll + 2];
        buf[0] = leftend;
        buf[sll + 1] = rightend;
        buf[1..=sll].copy_from_slice(&line[first..=lastpos]);
        erode_line_run(&mut buf, magnitude, extreme);
        line[first..=lastpos].copy_from_slice(&buf[1..=sll]);
    }
    line_dist.copy_from_slice(line);
    if last {
        for i in 0..n {
            line_lab[i] = if line[i] == base_sigma {
                line_lab[i]
            } else {
                0.0
            };
        }
    }
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_label_set_morph.rs"]
mod tests;
