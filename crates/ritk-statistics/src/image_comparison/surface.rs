use coeus_core::CpuAddressableStorage;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

/// Compute row-major strides for a shape given as a runtime slice.
pub(super) fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let d = shape.len();
    let mut strides = vec![1usize; d];
    let mut s = 1usize;
    for i in (0..d).rev() {
        strides[i] = s;
        s *= shape[i];
    }
    strides
}

/// Decode a flat row-major index to a multi-dimensional coordinate vector.
pub(super) fn flat_to_coords(mut flat: usize, strides: &[usize]) -> Vec<usize> {
    let d = strides.len();
    let mut coords = vec![0usize; d];
    for i in 0..d {
        coords[i] = flat / strides[i];
        flat %= strides[i];
    }
    coords
}

/// Encode a multi-dimensional coordinate vector to a flat row-major index.
#[inline]
pub(super) fn coords_to_flat(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides.iter())
        .map(|(&c, &s)| c * s)
        .sum()
}

/// Extract boundary voxels as physical coordinates from a flat binary mask.
///
/// A voxel is a boundary voxel if it is foreground (`value > 0.5`) and at
/// least one axis-aligned neighbor is background (`<= 0.5`) or out of bounds.
///
/// Returns `Vec<[f64; D]>` — one stack-sized array per boundary point — rather
/// than `Vec<Vec<f64>>`, eliminating the inner heap allocation per point.
///
/// Operates on the extracted `flat`/`shape` pair so the Coeus and Coeus-native
/// adapters share one boundary-extraction implementation.
pub(crate) fn extract_boundary_physical<const D: usize>(
    flat: &[f32],
    shape: [usize; D],
    spacing: &[f64; D],
) -> Vec<[f64; D]> {
    let shape_slice: &[usize] = &shape;
    let strides = compute_strides(shape_slice);
    let n_total: usize = shape_slice.iter().product();
    let mut boundary: Vec<[f64; D]> = Vec::with_capacity(n_total / 32);

    'voxel: for flat_idx in 0..n_total {
        if flat[flat_idx] <= crate::FOREGROUND_THRESHOLD {
            continue;
        }

        let coords = flat_to_coords(flat_idx, &strides);

        for dim in 0..D {
            for &delta in &[-1i64, 1i64] {
                let nb = coords[dim] as i64 + delta;
                if nb < 0 || nb >= shape[dim] as i64 {
                    let phys: [f64; D] = std::array::from_fn(|d| coords[d] as f64 * spacing[d]);
                    boundary.push(phys);
                    continue 'voxel;
                }
                let mut nb_coords = coords.clone();
                nb_coords[dim] = nb as usize;
                let nb_flat = coords_to_flat(&nb_coords, &strides);
                if flat[nb_flat] <= crate::FOREGROUND_THRESHOLD {
                    let phys: [f64; D] = std::array::from_fn(|d| coords[d] as f64 * spacing[d]);
                    boundary.push(phys);
                    continue 'voxel;
                }
            }
        }
    }

    boundary
}

/// Euclidean distance between two points of equal dimension.
#[inline]
pub(super) fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
        .sum::<f64>()
        .sqrt()
}

/// Minimum Euclidean distance from point `p` to any point in `set`.
///
/// Returns `f64::INFINITY` when `set` is empty.
#[inline]
pub(super) fn min_distance_to_set<const D: usize>(p: &[f64; D], set: &[[f64; D]]) -> f64 {
    set.iter()
        .map(|q| euclidean_distance(p, q))
        .fold(f64::INFINITY, f64::min)
}

/// Directed Hausdorff distance from `from_set` to `to_set`.
pub(super) fn directed_hausdorff<const D: usize>(
    from_set: &[[f64; D]],
    to_set: &[[f64; D]],
) -> f64 {
    if from_set.is_empty() {
        return 0.0;
    }
    from_set
        .iter()
        .map(|p| min_distance_to_set(p, to_set))
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Directed mean surface distance from `from_set` to `to_set`.
fn directed_msd<const D: usize>(from_set: &[[f64; D]], to_set: &[[f64; D]]) -> f64 {
    if from_set.is_empty() {
        return 0.0;
    }
    let total: f64 = from_set
        .iter()
        .map(|p| min_distance_to_set(p, to_set))
        .sum();
    total / from_set.len() as f64
}

/// Symmetric Hausdorff distance over two flat binary masks (shared host core).
///
/// The Coeus-backed [`hausdorff_distance`] and the Coeus-native
/// `native::hausdorff_distance` both extract `(flat, shape)` and delegate here,
/// so the boundary extraction and directed-distance math have exactly one home.
pub(crate) fn hausdorff_from_flat<const D: usize>(
    pred_flat: &[f32],
    pred_shape: [usize; D],
    gt_flat: &[f32],
    gt_shape: [usize; D],
    spacing: &[f64; D],
) -> f32 {
    let boundary_p = extract_boundary_physical(pred_flat, pred_shape, spacing);
    let boundary_g = extract_boundary_physical(gt_flat, gt_shape, spacing);

    if boundary_p.is_empty() && boundary_g.is_empty() {
        return 0.0;
    }

    let hd_p_to_g = directed_hausdorff(&boundary_p, &boundary_g);
    let hd_g_to_p = directed_hausdorff(&boundary_g, &boundary_p);

    hd_p_to_g.max(hd_g_to_p) as f32
}

/// Symmetric mean surface distance over two flat binary masks (shared host core).
pub(crate) fn msd_from_flat<const D: usize>(
    pred_flat: &[f32],
    pred_shape: [usize; D],
    gt_flat: &[f32],
    gt_shape: [usize; D],
    spacing: &[f64; D],
) -> f32 {
    let boundary_p = extract_boundary_physical(pred_flat, pred_shape, spacing);
    let boundary_g = extract_boundary_physical(gt_flat, gt_shape, spacing);

    if boundary_p.is_empty() && boundary_g.is_empty() {
        return 0.0;
    }

    let msd_p_to_g = directed_msd(&boundary_p, &boundary_g);
    let msd_g_to_p = directed_msd(&boundary_g, &boundary_p);

    ((msd_p_to_g + msd_g_to_p) / 2.0) as f32
}

/// Compute the Hausdorff distance between two binary segmentation masks.
pub fn hausdorff_distance<B: Backend, const D: usize>(
    prediction: &Image<f32, B, D>,
    ground_truth: &Image<f32, B, D>,
    spacing: &[f64; D],
) -> f32
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (pred_flat, pred_shape) = extract_vec_infallible(prediction);
    let (gt_flat, gt_shape) = extract_vec_infallible(ground_truth);
    hausdorff_from_flat(&pred_flat, pred_shape, &gt_flat, gt_shape, spacing)
}

/// Compute the symmetric mean surface distance between two binary masks.
pub fn mean_surface_distance<B: Backend, const D: usize>(
    prediction: &Image<f32, B, D>,
    ground_truth: &Image<f32, B, D>,
    spacing: &[f64; D],
) -> f32
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (pred_flat, pred_shape) = extract_vec_infallible(prediction);
    let (gt_flat, gt_shape) = extract_vec_infallible(ground_truth);
    msd_from_flat(&pred_flat, pred_shape, &gt_flat, gt_shape, spacing)
}
