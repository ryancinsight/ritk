//! Image comparison metrics for segmentation evaluation.
//!
//! Provides three spatial overlap and surface distance metrics:
//! - [`dice_coefficient`]: volumetric overlap between two binary masks.
//! - [`hausdorff_distance`]: maximum surface-to-surface distance.
//! - [`mean_surface_distance`]: symmetric mean surface-to-surface distance.
//!
//! # Mathematical Specification
//!
//! ## Dice Coefficient
//! Given binary masks P (prediction) and G (ground truth):
//!
//!   Dice(P, G) = 2·|P ∩ G| / (|P| + |G|)
//!
//! where |·| denotes the voxel count (sum of the binary mask).
//! Returns 1.0 when both masks are empty (trivially identical).
//!
//! ## Hausdorff Distance
//! Given boundary sets ∂P and ∂G in physical space (voxel indices × spacing):
//!
//!   HD(P, G) = max( max_{p ∈ ∂P} min_{g ∈ ∂G} d(p,g),
//!                   max_{g ∈ ∂G} min_{p ∈ ∂P} d(g,p) )
//!
//! where d is the Euclidean distance in physical coordinates.
//!
//! ## Mean Surface Distance
//! Directed MSD from set A to set B:
//!
//!   MSD(A→B) = (1/|A|) · Σ_{a ∈ A} min_{b ∈ B} d(a, b)
//!
//! Symmetric MSD:
//!
//!   MSD(P, G) = ( MSD(∂P → ∂G) + MSD(∂G → ∂P) ) / 2
//!
//! Returns 0.0 when both boundary sets are empty.

use crate::image::Image;
use burn::tensor::backend::Backend;

// ── Dice Coefficient ──────────────────────────────────────────────────────────

/// Compute the Dice similarity coefficient between two binary segmentation masks.
///
/// # Arguments
/// * `prediction`   – Binary mask (values 0.0 or 1.0).
/// * `ground_truth` – Binary mask (values 0.0 or 1.0).
///
/// # Returns
/// Dice score ∈ [0.0, 1.0].  Returns 1.0 when both masks are empty.
///
/// # Formula
/// `Dice = 2·|P ∩ G| / (|P| + |G|)`
pub fn dice_coefficient<B: Backend, const D: usize>(
    prediction: &Image<B, D>,
    ground_truth: &Image<B, D>,
) -> f32 {
    // Intersection via element-wise product: P ∩ G for binary masks = P * G.
    let intersection_sum: f32 = {
        let t = prediction.data().clone() * ground_truth.data().clone();
        let d = t.sum().into_data();
        d.as_slice::<f32>().expect("f32 intersection tensor")[0]
    };

    let pred_vol: f32 = {
        let d = prediction.data().clone().sum().into_data();
        d.as_slice::<f32>().expect("f32 prediction sum")[0]
    };

    let gt_vol: f32 = {
        let d = ground_truth.data().clone().sum().into_data();
        d.as_slice::<f32>().expect("f32 ground truth sum")[0]
    };

    let denom = pred_vol + gt_vol;
    if denom < f32::EPSILON {
        // Both masks are empty → trivially identical.
        return 1.0;
    }

    2.0 * intersection_sum / denom
}

// ── Boundary Extraction ───────────────────────────────────────────────────────

/// Compute row-major strides for a shape given as a runtime slice.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let d = shape.len();
    let mut strides = vec![1usize; d];
    let mut s = 1usize;
    for i in (0..d).rev() {
        strides[i] = s;
        s *= shape[i];
    }
    strides
}

/// Decode a flat (row-major) index to a multi-dimensional coordinate vector.
fn flat_to_coords(mut flat: usize, strides: &[usize]) -> Vec<usize> {
    let d = strides.len();
    let mut coords = vec![0usize; d];
    for i in 0..d {
        coords[i] = flat / strides[i];
        flat %= strides[i];
    }
    coords
}

/// Encode a multi-dimensional coordinate vector to a flat (row-major) index.
#[inline]
fn coords_to_flat(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides.iter())
        .map(|(&c, &s)| c * s)
        .sum()
}

/// Extract boundary voxels as physical coordinates from a binary mask.
///
/// A voxel is a boundary voxel if it is foreground (value > 0.5) and at least
/// one of its 2·D axis-aligned neighbors is background (≤ 0.5) or out-of-bounds.
///
/// Physical coordinate of voxel at multi-index `c` is `c[d] * spacing[d]` for
/// each dimension `d`.
///
/// # Returns
/// Each element of the returned `Vec` is a `Vec<f64>` of length D representing
/// the physical coordinates of one boundary voxel.
fn extract_boundary_physical<B: Backend, const D: usize>(
    mask: &Image<B, D>,
    spacing: &[f64; D],
) -> Vec<Vec<f64>> {
    let shape: [usize; D] = mask.shape();
    let shape_slice: &[usize] = &shape;

    let mask_tensor_data = mask.data().clone().into_data();
    let flat = mask_tensor_data
        .as_slice::<f32>()
        .expect("f32 mask tensor data");

    let strides = compute_strides(shape_slice);
    let n_total: usize = shape_slice.iter().product();

    let mut boundary: Vec<Vec<f64>> = Vec::new();

    'voxel: for flat_idx in 0..n_total {
        if flat[flat_idx] <= 0.5 {
            continue;
        }

        let coords = flat_to_coords(flat_idx, &strides);

        // Check all 2·D axis-aligned neighbors.
        for dim in 0..D {
            for &delta in &[-1i64, 1i64] {
                let nb = coords[dim] as i64 + delta;
                if nb < 0 || nb >= shape[dim] as i64 {
                    // Out-of-bounds → surface voxel.
                    let phys: Vec<f64> = (0..D).map(|d| coords[d] as f64 * spacing[d]).collect();
                    boundary.push(phys);
                    continue 'voxel;
                }
                let mut nb_coords = coords.clone();
                nb_coords[dim] = nb as usize;
                let nb_flat = coords_to_flat(&nb_coords, &strides);
                if flat[nb_flat] <= 0.5 {
                    // Adjacent to background → surface voxel.
                    let phys: Vec<f64> = (0..D).map(|d| coords[d] as f64 * spacing[d]).collect();
                    boundary.push(phys);
                    continue 'voxel;
                }
            }
        }
    }

    boundary
}

// ── Distance Primitives ───────────────────────────────────────────────────────

/// Euclidean distance between two points of equal dimension.
#[inline]
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
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
fn min_distance_to_set(p: &[f64], set: &[Vec<f64>]) -> f64 {
    set.iter()
        .map(|q| euclidean_distance(p, q))
        .fold(f64::INFINITY, f64::min)
}

/// Directed Hausdorff distance from `from_set` to `to_set`.
///
/// `max_{p ∈ from_set} min_{q ∈ to_set} d(p, q)`
///
/// Returns 0.0 when `from_set` is empty; returns +∞ when `to_set` is empty
/// and `from_set` is non-empty.
fn directed_hausdorff(from_set: &[Vec<f64>], to_set: &[Vec<f64>]) -> f64 {
    if from_set.is_empty() {
        return 0.0;
    }
    from_set
        .iter()
        .map(|p| min_distance_to_set(p, to_set))
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Directed mean surface distance from `from_set` to `to_set`.
///
/// `(1/|from_set|) · Σ_{p ∈ from_set} min_{q ∈ to_set} d(p, q)`
///
/// Returns 0.0 when `from_set` is empty; returns +∞ when `to_set` is empty
/// and `from_set` is non-empty.
fn directed_msd(from_set: &[Vec<f64>], to_set: &[Vec<f64>]) -> f64 {
    if from_set.is_empty() {
        return 0.0;
    }
    let total: f64 = from_set
        .iter()
        .map(|p| min_distance_to_set(p, to_set))
        .sum();
    total / from_set.len() as f64
}

// ── Public Surface Distance APIs ──────────────────────────────────────────────

/// Compute the Hausdorff distance between two binary segmentation masks.
///
/// # Arguments
/// * `prediction`   – Binary mask (values 0.0 or 1.0).
/// * `ground_truth` – Binary mask (values 0.0 or 1.0).
/// * `spacing`      – Physical voxel spacing in each dimension.
///
/// # Returns
/// Symmetric Hausdorff distance in the same units as `spacing`.
/// Returns 0.0 when both boundary sets are empty.
///
/// # Formula
/// `HD = max( HD(∂P→∂G), HD(∂G→∂P) )`
pub fn hausdorff_distance<B: Backend, const D: usize>(
    prediction: &Image<B, D>,
    ground_truth: &Image<B, D>,
    spacing: &[f64; D],
) -> f32 {
    let boundary_p = extract_boundary_physical(prediction, spacing);
    let boundary_g = extract_boundary_physical(ground_truth, spacing);

    if boundary_p.is_empty() && boundary_g.is_empty() {
        return 0.0;
    }

    let hd_p_to_g = directed_hausdorff(&boundary_p, &boundary_g);
    let hd_g_to_p = directed_hausdorff(&boundary_g, &boundary_p);

    hd_p_to_g.max(hd_g_to_p) as f32
}

/// Compute the symmetric mean surface distance between two binary segmentation masks.
///
/// # Arguments
/// * `prediction`   – Binary mask (values 0.0 or 1.0).
/// * `ground_truth` – Binary mask (values 0.0 or 1.0).
/// * `spacing`      – Physical voxel spacing in each dimension.
///
/// # Returns
/// Symmetric mean surface distance in the same units as `spacing`.
/// Returns 0.0 when both boundary sets are empty.
///
/// # Formula
/// `MSD = ( MSD(∂P→∂G) + MSD(∂G→∂P) ) / 2`
pub fn mean_surface_distance<B: Backend, const D: usize>(
    prediction: &Image<B, D>,
    ground_truth: &Image<B, D>,
    spacing: &[f64; D],
) -> f32 {
    let boundary_p = extract_boundary_physical(prediction, spacing);
    let boundary_g = extract_boundary_physical(ground_truth, spacing);

    if boundary_p.is_empty() && boundary_g.is_empty() {
        return 0.0;
    }

    let msd_p_to_g = directed_msd(&boundary_p, &boundary_g);
    let msd_g_to_p = directed_msd(&boundary_g, &boundary_p);

    ((msd_p_to_g + msd_g_to_p) / 2.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn make_mask_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn make_mask_2d(data: Vec<f32>, dims: [usize; 2]) -> Image<TestBackend, 2> {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(data, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0]),
            Spacing::new([1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn make_mask_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
        let n = data.len();
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 1>::from_data(TensorData::new(data, Shape::new([n])), &device);
        Image::new(
            tensor,
            Point::new([0.0]),
            Spacing::new([1.0]),
            Direction::identity(),
        )
    }

    // ── Dice: positive tests ──────────────────────────────────────────────────

    #[test]
    fn test_dice_identical_masks_is_one() {
        // All-foreground 3x3x3 mask against itself → Dice = 1.0.
        let mask = make_mask_3d(vec![1.0f32; 27], [3, 3, 3]);
        let dice = dice_coefficient(&mask, &mask);
        assert!(
            (dice - 1.0).abs() < 1e-5,
            "identical masks → Dice = 1.0, got {}",
            dice
        );
    }

    #[test]
    fn test_dice_disjoint_masks_is_zero() {
        // Non-overlapping masks: intersection = 0 → Dice = 0.
        // P: indices 0..13, G: indices 14..27.
        let mut pred = vec![0.0f32; 27];
        for i in 0..13 {
            pred[i] = 1.0;
        }
        let mut gt = vec![0.0f32; 27];
        for i in 14..27 {
            gt[i] = 1.0;
        }
        let pred_img = make_mask_3d(pred, [3, 3, 3]);
        let gt_img = make_mask_3d(gt, [3, 3, 3]);
        let dice = dice_coefficient(&pred_img, &gt_img);
        assert!(
            dice.abs() < 1e-5,
            "disjoint masks → Dice = 0.0, got {}",
            dice
        );
    }

    #[test]
    fn test_dice_known_overlap_half() {
        // P = {0,1,2,3}, G = {2,3,4,5} in a length-8 1D mask.
        // |P ∩ G| = |{2,3}| = 2, |P| = 4, |G| = 4.
        // Dice = 2·2 / (4+4) = 0.5.
        let pred = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let gt = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let pred_img = make_mask_1d(pred);
        let gt_img = make_mask_1d(gt);
        let dice = dice_coefficient(&pred_img, &gt_img);
        assert!(
            (dice - 0.5).abs() < 1e-5,
            "Dice = 2·2/(4+4) = 0.5, got {}",
            dice
        );
    }

    #[test]
    fn test_dice_both_empty_returns_one() {
        // Both all-zero → denominator = 0 → returns 1.0 by convention.
        let pred = make_mask_3d(vec![0.0; 27], [3, 3, 3]);
        let gt = make_mask_3d(vec![0.0; 27], [3, 3, 3]);
        let dice = dice_coefficient(&pred, &gt);
        assert!(
            (dice - 1.0).abs() < 1e-5,
            "both empty → Dice = 1.0, got {}",
            dice
        );
    }

    #[test]
    fn test_dice_2d_known_overlap() {
        // 4×4 masks.  P: rows 0..2, cols 0..2 (4 voxels).
        //              G: rows 0..2, cols 1..3 (4 voxels).
        // Overlap column 1 across rows 0,1 → |P ∩ G| = 2.
        // Dice = 2·2/(4+4) = 0.5.
        let mut pred = vec![0.0f32; 16];
        let mut gt = vec![0.0f32; 16];
        // Row 0: pred cols 0,1 ; gt cols 1,2
        pred[0] = 1.0;
        pred[1] = 1.0;
        gt[1] = 1.0;
        gt[2] = 1.0;
        // Row 1: pred cols 0,1 ; gt cols 1,2  (offset +4)
        pred[4] = 1.0;
        pred[5] = 1.0;
        gt[5] = 1.0;
        gt[6] = 1.0;

        let pred_img = make_mask_2d(pred, [4, 4]);
        let gt_img = make_mask_2d(gt, [4, 4]);
        let dice = dice_coefficient(&pred_img, &gt_img);
        assert!((dice - 0.5).abs() < 1e-5, "2D Dice = 0.5, got {}", dice);
    }

    #[test]
    fn test_dice_symmetry() {
        // Dice(P, G) = Dice(G, P).
        let pred = make_mask_1d(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let gt = make_mask_1d(vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        let d_pg = dice_coefficient(&pred, &gt);
        let d_gp = dice_coefficient(&gt, &pred);
        assert!(
            (d_pg - d_gp).abs() < 1e-6,
            "Dice is not symmetric: {} vs {}",
            d_pg,
            d_gp
        );
    }

    // ── Dice: boundary / negative ─────────────────────────────────────────────

    #[test]
    fn test_dice_one_empty_one_nonempty_is_zero() {
        // |P| = 0, |G| = k > 0 → denominator = k → Dice = 0.
        let pred = make_mask_1d(vec![0.0; 8]);
        let gt = make_mask_1d(vec![1.0; 8]);
        let dice = dice_coefficient(&pred, &gt);
        assert!(dice.abs() < 1e-5, "one empty → Dice = 0.0, got {}", dice);
    }

    // ── Hausdorff distance: positive tests ────────────────────────────────────

    #[test]
    fn test_hausdorff_identical_masks_is_zero() {
        // Every boundary voxel of P is also on G; min distance = 0 → HD = 0.
        let mask = make_mask_2d(vec![1.0f32; 9], [3, 3]);
        let spacing = [1.0f64, 1.0];
        let hd = hausdorff_distance(&mask, &mask, &spacing);
        assert!(hd.abs() < 1e-5, "identical masks → HD = 0.0, got {}", hd);
    }

    #[test]
    fn test_hausdorff_1d_known_value() {
        // P = [1,1,0,0,0], G = [0,0,0,1,1], spacing = 1.0.
        // ∂P = {0, 1},  ∂G = {3, 4}  (all foreground voxels are boundary in 1D).
        // Physical coords: ∂P = {0.0, 1.0}, ∂G = {3.0, 4.0}.
        // HD(P→G) = max( min_d(0→{3,4}), min_d(1→{3,4}) ) = max(3, 2) = 3.
        // HD(G→P) = max( min_d(3→{0,1}), min_d(4→{0,1}) ) = max(2, 3) = 3.
        // HD = 3.0.
        let pred = make_mask_1d(vec![1.0, 1.0, 0.0, 0.0, 0.0]);
        let gt = make_mask_1d(vec![0.0, 0.0, 0.0, 1.0, 1.0]);
        let spacing = [1.0f64];
        let hd = hausdorff_distance(&pred, &gt, &spacing);
        assert!((hd - 3.0).abs() < 1e-4, "1D HD expected 3.0, got {}", hd);
    }

    #[test]
    fn test_hausdorff_scales_with_spacing() {
        // Same geometry as above but spacing = 2.0 → all distances × 2 → HD = 6.0.
        let device: <TestBackend as Backend>::Device = Default::default();
        let pred_t = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![1.0f32, 1.0, 0.0, 0.0, 0.0], Shape::new([5])),
            &device,
        );
        let gt_t = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 0.0, 1.0, 1.0], Shape::new([5])),
            &device,
        );
        let pred_img: Image<TestBackend, 1> = Image::new(
            pred_t,
            Point::new([0.0]),
            Spacing::new([2.0]),
            Direction::identity(),
        );
        let gt_img: Image<TestBackend, 1> = Image::new(
            gt_t,
            Point::new([0.0]),
            Spacing::new([2.0]),
            Direction::identity(),
        );
        let spacing = [2.0f64];
        let hd = hausdorff_distance(&pred_img, &gt_img, &spacing);
        assert!(
            (hd - 6.0).abs() < 1e-4,
            "spacing=2.0 → HD = 6.0, got {}",
            hd
        );
    }

    #[test]
    fn test_hausdorff_symmetry() {
        // HD(P, G) = HD(G, P).
        let pred = make_mask_1d(vec![1.0, 1.0, 0.0, 0.0, 0.0]);
        let gt = make_mask_1d(vec![0.0, 0.0, 0.0, 1.0, 1.0]);
        let spacing = [1.0f64];
        let hd_pg = hausdorff_distance(&pred, &gt, &spacing);
        let hd_gp = hausdorff_distance(&gt, &pred, &spacing);
        assert!(
            (hd_pg - hd_gp).abs() < 1e-5,
            "HD not symmetric: {} vs {}",
            hd_pg,
            hd_gp
        );
    }

    #[test]
    fn test_hausdorff_both_empty_is_zero() {
        let pred = make_mask_3d(vec![0.0; 27], [3, 3, 3]);
        let gt = make_mask_3d(vec![0.0; 27], [3, 3, 3]);
        let spacing = [1.0f64, 1.0, 1.0];
        let hd = hausdorff_distance(&pred, &gt, &spacing);
        assert!(hd.abs() < 1e-5, "both empty → HD = 0.0, got {}", hd);
    }

    // ── Mean surface distance: positive tests ─────────────────────────────────

    #[test]
    fn test_msd_identical_masks_is_zero() {
        // Every boundary voxel of P coincides with a boundary voxel of G → MSD = 0.
        let mask = make_mask_2d(vec![1.0f32; 9], [3, 3]);
        let spacing = [1.0f64, 1.0];
        let msd = mean_surface_distance(&mask, &mask, &spacing);
        assert!(msd.abs() < 1e-5, "identical masks → MSD = 0.0, got {}", msd);
    }

    #[test]
    fn test_msd_1d_known_value() {
        // P = [1,1,0,0,0], G = [0,0,0,1,1], spacing = 1.0.
        // ∂P = {0.0, 1.0}, ∂G = {3.0, 4.0}.
        // MSD(P→G): (min_d(0,{3,4}) + min_d(1,{3,4})) / 2 = (3 + 2) / 2 = 2.5.
        // MSD(G→P): (min_d(3,{0,1}) + min_d(4,{0,1})) / 2 = (2 + 3) / 2 = 2.5.
        // Symmetric MSD = (2.5 + 2.5) / 2 = 2.5.
        let pred = make_mask_1d(vec![1.0, 1.0, 0.0, 0.0, 0.0]);
        let gt = make_mask_1d(vec![0.0, 0.0, 0.0, 1.0, 1.0]);
        let spacing = [1.0f64];
        let msd = mean_surface_distance(&pred, &gt, &spacing);
        assert!((msd - 2.5).abs() < 1e-4, "1D MSD expected 2.5, got {}", msd);
    }

    #[test]
    fn test_msd_leq_hausdorff() {
        // By definition MSD ≤ HD (mean ≤ max of the same per-point distances).
        let pred = make_mask_1d(vec![1.0, 1.0, 0.0, 0.0, 0.0]);
        let gt = make_mask_1d(vec![0.0, 0.0, 0.0, 1.0, 1.0]);
        let spacing = [1.0f64];
        let hd = hausdorff_distance(&pred, &gt, &spacing);
        let msd = mean_surface_distance(&pred, &gt, &spacing);
        assert!(msd <= hd + 1e-5, "MSD ({}) must be ≤ HD ({})", msd, hd);
    }

    #[test]
    fn test_msd_both_empty_is_zero() {
        let pred = make_mask_3d(vec![0.0; 27], [3, 3, 3]);
        let gt = make_mask_3d(vec![0.0; 27], [3, 3, 3]);
        let spacing = [1.0f64, 1.0, 1.0];
        let msd = mean_surface_distance(&pred, &gt, &spacing);
        assert!(msd.abs() < 1e-5, "both empty → MSD = 0.0, got {}", msd);
    }

    #[test]
    fn test_msd_symmetry() {
        // MSD(P, G) = MSD(G, P).
        let pred = make_mask_1d(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let gt = make_mask_1d(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
        let spacing = [1.0f64];
        let msd_pg = mean_surface_distance(&pred, &gt, &spacing);
        let msd_gp = mean_surface_distance(&gt, &pred, &spacing);
        assert!(
            (msd_pg - msd_gp).abs() < 1e-5,
            "MSD is not symmetric: {} vs {}",
            msd_pg,
            msd_gp
        );
    }

    // ── Utility function unit tests ───────────────────────────────────────────

    #[test]
    fn test_strides_3d() {
        // Shape [2, 3, 4] → strides [12, 4, 1].
        let shape = [2usize, 3, 4];
        let strides = compute_strides(&shape);
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_flat_to_coords_round_trip() {
        // For shape [3, 4, 5], flat index 37 → (1, 3, 2) → back to 37.
        let shape = [3usize, 4, 5];
        let strides = compute_strides(&shape);
        let coords = flat_to_coords(37, &strides);
        assert_eq!(coords, vec![1, 3, 2]);
        let back = coords_to_flat(&coords, &strides);
        assert_eq!(back, 37);
    }

    #[test]
    fn test_euclidean_distance_known() {
        // d((0,0), (3,4)) = 5.
        let a = vec![0.0f64, 0.0];
        let b = vec![3.0f64, 4.0];
        let d = euclidean_distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-10, "expected 5.0, got {}", d);
    }

    #[test]
    fn test_min_distance_to_empty_set_is_infinity() {
        let p = vec![1.0f64, 2.0];
        let empty: Vec<Vec<f64>> = vec![];
        let d = min_distance_to_set(&p, &empty);
        assert!(d.is_infinite() && d > 0.0, "empty set → +∞");
    }

    #[test]
    fn test_directed_hausdorff_empty_from_is_zero() {
        let from: Vec<Vec<f64>> = vec![];
        let to = vec![vec![1.0f64, 0.0], vec![2.0, 0.0]];
        assert_eq!(directed_hausdorff(&from, &to), 0.0);
    }
}
