use anyhow::{anyhow, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Shape2d {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Shape3d {
    pub(super) depth: usize,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) len: usize,
}

/// Padded shape for a `D`-dimensional FFT buffer.
///
/// Rank-generic so the 2-D and 3-D convolution/NCC paths share one
/// non-aliasing extent calculation (see [`checked_fft_shape`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct FftShape<const D: usize> {
    /// Per-axis padded extents, outermost axis first.
    pub(super) dims: [usize; D],
    /// Total element count, i.e. `product(dims)`.
    pub(super) len: usize,
}

pub(super) fn checked_edge_shape_2d(
    [rows, cols]: [usize; 2],
    [row_radius, col_radius]: [usize; 2],
    context: &str,
) -> Result<Shape2d> {
    require_nonzero::<2>([rows, cols], context, "input")?;
    let padded_rows = checked_radius_pad(rows, row_radius, context, "rows")?;
    let padded_cols = checked_radius_pad(cols, col_radius, context, "cols")?;
    shape_2d([padded_rows, padded_cols], context, "edge-padded buffer")
}

pub(super) fn checked_edge_shape_3d(
    [depth, rows, cols]: [usize; 3],
    [depth_radius, row_radius, col_radius]: [usize; 3],
    context: &str,
) -> Result<Shape3d> {
    require_nonzero::<3>([depth, rows, cols], context, "input")?;
    let padded_depth = checked_radius_pad(depth, depth_radius, context, "depth")?;
    let padded_rows = checked_radius_pad(rows, row_radius, context, "rows")?;
    let padded_cols = checked_radius_pad(cols, col_radius, context, "cols")?;
    shape_3d(
        [padded_depth, padded_rows, padded_cols],
        context,
        "edge-padded buffer",
    )
}

/// Next-power-of-two FFT padding for a `D`-D linear convolution.
///
/// Each axis is padded to `next_power_of_two(input + kernel − 1)` so the
/// circular FFT convolution equals the linear one (no wraparound aliasing).
/// `D` must be `2` or `3`.
pub(super) fn checked_fft_shape<const D: usize>(
    input: [usize; D],
    kernel: [usize; D],
    context: &str,
) -> Result<FftShape<D>> {
    require_nonzero::<D>(input, context, "input")?;
    require_nonzero::<D>(kernel, context, "kernel")?;
    let mut dims = [0usize; D];
    for axis in 0..D {
        dims[axis] = checked_fft_extent(input[axis], kernel[axis], context, axis_name::<D>(axis))?;
    }
    let len = fft_shape_len::<D>(dims, context)?;
    Ok(FftShape { dims, len })
}

pub(super) fn edge_source_index(padded_index: usize, radius: usize, extent: usize) -> usize {
    debug_assert!(extent > 0);
    padded_index.saturating_sub(radius).min(extent - 1)
}

/// Diagnostic axis label for `axis` in a `D`-D shape (outermost axis first).
fn axis_name<const D: usize>(axis: usize) -> &'static str {
    match D - axis {
        1 => "cols",
        2 => "rows",
        _ => "depth",
    }
}

/// Total element count of an FFT buffer, preserving the axis-specific overflow
/// diagnostics the previous 2-D and 3-D helpers produced.
fn fft_shape_len<const D: usize>(dims: [usize; D], context: &str) -> Result<usize> {
    match D {
        2 => dims[0].checked_mul(dims[1]).ok_or_else(|| {
            anyhow!(
                "{context}: FFT buffer element count {} * {} overflows usize",
                dims[0],
                dims[1]
            )
        }),
        3 => {
            let slice = dims[1].checked_mul(dims[2]).ok_or_else(|| {
                anyhow!(
                    "{context}: FFT buffer slice element count {} * {} overflows usize",
                    dims[1],
                    dims[2]
                )
            })?;
            dims[0].checked_mul(slice).ok_or_else(|| {
                anyhow!(
                    "{context}: FFT buffer element count {} * {slice} overflows usize",
                    dims[0]
                )
            })
        }
        _ => Err(anyhow!(
            "{context}: FFT padding supports only 2-D and 3-D, got rank {D}"
        )),
    }
}

fn checked_radius_pad(
    extent: usize,
    radius: usize,
    context: &str,
    axis_name: &str,
) -> Result<usize> {
    let diameter = radius.checked_mul(2).ok_or_else(|| {
        anyhow!("{context}: {axis_name} padding radius {radius} overflows usize when doubled")
    })?;
    extent.checked_add(diameter).ok_or_else(|| {
        anyhow!(
            "{context}: {axis_name} extent {extent} plus boundary padding {diameter} overflows usize"
        )
    })
}

fn checked_fft_extent(
    extent: usize,
    kernel_extent: usize,
    context: &str,
    axis_name: &str,
) -> Result<usize> {
    let linear_extent = extent
        .checked_add(kernel_extent)
        .and_then(|sum| sum.checked_sub(1))
        .ok_or_else(|| {
            anyhow!(
                "{context}: {axis_name} linear extent {extent} + {kernel_extent} - 1 overflows usize"
            )
        })?;
    linear_extent.checked_next_power_of_two().ok_or_else(|| {
        anyhow!(
            "{context}: {axis_name} linear extent {linear_extent} has no representable power-of-two FFT padding"
        )
    })
}

fn shape_2d([rows, cols]: [usize; 2], context: &str, role: &str) -> Result<Shape2d> {
    let len = rows.checked_mul(cols).ok_or_else(|| {
        anyhow!("{context}: {role} element count {rows} * {cols} overflows usize")
    })?;
    Ok(Shape2d { rows, cols, len })
}

fn shape_3d([depth, rows, cols]: [usize; 3], context: &str, role: &str) -> Result<Shape3d> {
    let slice_len = rows.checked_mul(cols).ok_or_else(|| {
        anyhow!("{context}: {role} slice element count {rows} * {cols} overflows usize")
    })?;
    let len = depth.checked_mul(slice_len).ok_or_else(|| {
        anyhow!("{context}: {role} element count {depth} * {slice_len} overflows usize")
    })?;
    Ok(Shape3d {
        depth,
        rows,
        cols,
        len,
    })
}

/// Reject any zero-length axis in `shape`.
fn require_nonzero<const D: usize>(shape: [usize; D], context: &str, role: &str) -> Result<()> {
    if shape.iter().all(|&extent| extent > 0) {
        return Ok(());
    }
    let dims = shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    Err(anyhow!(
        "{context}: {role} dimensions must be non-zero, got [{dims}]"
    ))
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
    use super::{
        checked_edge_shape_2d, checked_edge_shape_3d, checked_fft_shape, edge_source_index,
    };

    #[test]
    fn checked_fft_shape_2d_uses_linear_convolution_extent() {
        let shape = checked_fft_shape::<2>([5, 6], [3, 4], "fft2").unwrap();

        assert_eq!((shape.dims[0], shape.dims[1], shape.len), (8, 16, 128));
    }

    #[test]
    fn checked_fft_shape_3d_tracks_slice_and_total_len() {
        let shape = checked_fft_shape::<3>([5, 6, 7], [3, 4, 5], "fft3").unwrap();
        let [depth, rows, cols] = shape.dims;
        let slice_len = rows * cols;

        assert_eq!(
            (depth, rows, cols, slice_len, shape.len),
            (8, 16, 16, 256, 2048)
        );
    }

    #[test]
    fn checked_edge_shape_2d_rejects_radius_overflow() {
        let error = checked_edge_shape_2d([8, 8], [usize::MAX, 0], "edge2").unwrap_err();

        assert_eq!(
            error.to_string(),
            format!(
                "edge2: rows padding radius {} overflows usize when doubled",
                usize::MAX
            )
        );
    }

    #[test]
    fn checked_fft_shape_2d_rejects_extent_without_power_of_two() {
        let oversized_extent = usize::MAX / 2 + 2;
        let error = checked_fft_shape::<2>([oversized_extent, 8], [1, 1], "fft2").unwrap_err();

        assert_eq!(
            error.to_string(),
            format!(
                "fft2: rows linear extent {} has no representable power-of-two FFT padding",
                oversized_extent
            )
        );
    }

    #[test]
    fn checked_fft_shape_3d_rejects_total_len_overflow() {
        let oversized_depth = usize::MAX / 4;
        let padded_depth = oversized_depth.checked_next_power_of_two().unwrap();
        let error = checked_fft_shape::<3>([oversized_depth, 2, 2], [1, 1, 1], "fft3").unwrap_err();

        assert_eq!(
            error.to_string(),
            format!("fft3: FFT buffer element count {padded_depth} * 4 overflows usize")
        );
    }

    #[test]
    fn checked_edge_shape_3d_rejects_zero_input() {
        let error = checked_edge_shape_3d([0, 4, 4], [0, 0, 0], "edge3").unwrap_err();

        assert_eq!(
            error.to_string(),
            "edge3: input dimensions must be non-zero, got [0, 4, 4]"
        );
    }

    #[test]
    fn edge_source_index_clamps_to_valid_input_extent() {
        assert_eq!(edge_source_index(0, 2, 5), 0);
        assert_eq!(edge_source_index(3, 2, 5), 1);
        assert_eq!(edge_source_index(8, 2, 5), 4);
    }
}
