use super::key::F32Key;
use super::map::ValueIndices;
use ritk_image::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;
use std::collections::HashMap;

/// Build the per-value index map for an image.
///
/// Equivalent to `scipy.ndimage.value_indices(arr, ignore_value=…)`:
/// returns a dictionary mapping each distinct voxel value to the
/// multi-indices where it occurs, in row-major order.
///
/// # Parameters
///
/// - `image`: any D-dimensional image backed by `f32` storage
///   (mirrors the contract of the other `ritk_statistics`
///   routines that use `extract_vec_infallible`).
/// - `ignore_value`: if `Some(v)`, voxels equal to `v` (by bit
///   pattern) are excluded from the output. `None` includes every
///   distinct value.
///
/// # Examples
///
/// ```ignore
/// // 3-D image with two distinct non-zero values.
/// let img = Image::<f32, 3>::from_vec_f32(
///     [2, 2, 2],
///     vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
/// )?;
/// let vi = value_indices(&img, None);
/// assert_eq!(vi.len(1.0), 4);
/// assert_eq!(vi.len(2.0), 4);
/// // 1.0 occurs at (0,0,0), (0,1,0), (1,0,0), (1,1,0) in row-major.
/// assert_eq!(vi.get(1.0), Some(&[[0,0,0], [0,1,0], [1,0,0], [1,1,0]]));
/// ```
///
/// # Panics
///
/// Panics if the underlying backend's tensor data cannot be extracted
/// as `Vec<f32>` (only possible with non-`f32` backends; see
/// `extract_vec_infallible`).
pub fn value_indices<B: Backend, const D: usize>(
    image: &Image<B, D>,
    ignore_value: Option<f32>,
) -> ValueIndices<D> {
    let (vals, dims) = extract_vec_infallible(image);

    let ignore_key = ignore_value.map(F32Key::new);

    // Pre-allocate to avoid rehashing for typical medical images with
    // few distinct labels. When an ignore_value is specified (common
    // case: background label), roughly half the voxel values are
    // expected to be distinct; otherwise ~25% is a conservative
    // estimate for natural images.
    let capacity_estimate = if ignore_value.is_some() {
        vals.len() / 2
    } else {
        vals.len() / 4
    };
    let mut indices: HashMap<F32Key, Vec<[usize; D]>> = HashMap::with_capacity(capacity_estimate);

    for (flat, &v) in vals.iter().enumerate() {
        let key = F32Key::new(v);
        if Some(key) == ignore_key {
            continue;
        }
        indices
            .entry(key)
            .or_default()
            .push(flat_to_multi(flat, dims));
    }

    ValueIndices { indices }
}

/// Convert a flat row-major index to a multi-index `[i_0, i_1, …, i_{D-1}]`.
///
/// Row-major: `p = i_0 · (d_1 · d_2 · …) + i_1 · (d_2 · …) + … + i_{D-1}`.
///
/// Matches the layout produced by `Image::from_vec_f32` and
/// `extract_vec_infallible`. Equivalent to the private helper of the
/// same name in `position_extrema`; duplicated here to keep modules
/// independent (each is a leaf under the `statistics` bounded context).
#[inline]
pub(crate) fn flat_to_multi<const D: usize>(flat: usize, dims: [usize; D]) -> [usize; D] {
    let mut out = [0_usize; D];
    let mut remaining = flat;
    for (k, out_k) in out.iter_mut().enumerate() {
        let mut stride: usize = 1;
        for &d in dims.iter().skip(k + 1) {
            stride *= d;
        }
        *out_k = remaining / stride;
        remaining %= stride;
    }
    out
}
