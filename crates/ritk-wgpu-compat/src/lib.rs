//! WGPU dispatch-limit constants.
//!
//! WGPU limits individual dispatch dimensions to 65 535 invocations.
//! All point-batch and tensor-row operations that may exceed this limit
//! should respect these chunk ceilings.
//!
//! [`apply_row_chunks`] owns chunk scheduling without depending on a tensor
//! provider. Callers supply their own slice, operation, and concatenation
//! closures, so the same WGPU limit policy serves native and legacy backends.

use std::ops::Range;

/// Maximum rows dispatched per shader invocation for 2-D/3-D operations.
///
/// Half the WGPU hard-limit (65 535) provides a conservative safe margin.
pub const WGPU_CHUNK_SIZE: usize = 32_768;

/// Maximum rows dispatched per shader invocation for the 4-D B-spline path.
///
/// The 4-D kernel issues more sub-dispatches per point than the 3-D path,
/// so a smaller ceiling is required to stay inside the WGPU limit.
pub const WGPU_CHUNK_SIZE_4D: usize = 16_384;

/// Apply an operation to a row-batched value without exceeding a WGPU dispatch
/// limit.
///
/// `rows` describes the leading row axis of `value`. For an input no larger
/// than `chunk_size`, `apply` receives ownership of `value` directly. Larger
/// inputs are sliced by `slice`, operated independently, then reassembled by
/// `concat`; no tensor provider or element type enters this scheduling layer.
///
/// # Panics
///
/// Panics when `chunk_size` is zero. The WGPU limits exported by this crate are
/// positive constants, so this signals a caller configuration defect.
pub fn apply_row_chunks<T, Apply, Slice, Concat>(
    value: T,
    rows: usize,
    chunk_size: usize,
    apply: Apply,
    slice: Slice,
    concat: Concat,
) -> T
where
    Apply: Fn(T) -> T,
    Slice: Fn(&T, Range<usize>) -> T,
    Concat: FnOnce(Vec<T>) -> T,
{
    assert!(chunk_size > 0, "WGPU row chunk size must be positive");
    if rows <= chunk_size {
        return apply(value);
    }

    let mut chunks = Vec::with_capacity(rows.div_ceil(chunk_size));
    for start in (0..rows).step_by(chunk_size) {
        let end = (start + chunk_size).min(rows);
        chunks.push(apply(slice(&value, start..end)));
    }
    concat(chunks)
}

#[cfg(test)]
mod tests {
    use super::apply_row_chunks;

    #[test]
    fn row_chunk_apply_preserves_small_values_without_slicing() {
        let output = apply_row_chunks(
            vec![1, 2, 3],
            3,
            8,
            |value| value,
            |_, _| panic!("small input must not be sliced"),
            |_| panic!("small input must not be concatenated"),
        );

        assert_eq!(output, vec![1, 2, 3]);
    }

    #[test]
    fn row_chunk_apply_preserves_uneven_chunk_order() {
        let output = apply_row_chunks(
            (0..20).collect::<Vec<_>>(),
            20,
            8,
            |chunk| chunk.into_iter().map(|value| value * 2).collect(),
            |value, range| value[range].to_vec(),
            |chunks| chunks.into_iter().flatten().collect(),
        );

        assert_eq!(output, (0..20).map(|value| value * 2).collect::<Vec<_>>());
    }

    #[test]
    #[should_panic(expected = "WGPU row chunk size must be positive")]
    fn row_chunk_apply_rejects_zero_chunk_size() {
        let _ = apply_row_chunks(
            vec![1],
            1,
            0,
            |value| value,
            |_, _| Vec::new(),
            |chunks| chunks.into_iter().flatten().collect(),
        );
    }
}
