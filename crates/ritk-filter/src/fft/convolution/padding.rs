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
    pub(super) slice_len: usize,
    pub(super) len: usize,
}

pub(super) fn checked_edge_shape_2d(
    [rows, cols]: [usize; 2],
    [row_radius, col_radius]: [usize; 2],
    context: &str,
) -> Result<Shape2d> {
    require_nonzero_2d([rows, cols], context, "input")?;
    let padded_rows = checked_radius_pad(rows, row_radius, context, "rows")?;
    let padded_cols = checked_radius_pad(cols, col_radius, context, "cols")?;
    shape_2d([padded_rows, padded_cols], context, "edge-padded buffer")
}

pub(super) fn checked_edge_shape_3d(
    [depth, rows, cols]: [usize; 3],
    [depth_radius, row_radius, col_radius]: [usize; 3],
    context: &str,
) -> Result<Shape3d> {
    require_nonzero_3d([depth, rows, cols], context, "input")?;
    let padded_depth = checked_radius_pad(depth, depth_radius, context, "depth")?;
    let padded_rows = checked_radius_pad(rows, row_radius, context, "rows")?;
    let padded_cols = checked_radius_pad(cols, col_radius, context, "cols")?;
    shape_3d(
        [padded_depth, padded_rows, padded_cols],
        context,
        "edge-padded buffer",
    )
}

pub(super) fn checked_fft_shape_2d(
    input: [usize; 2],
    kernel: [usize; 2],
    context: &str,
) -> Result<Shape2d> {
    require_nonzero_2d(input, context, "input")?;
    require_nonzero_2d(kernel, context, "kernel")?;
    let rows = checked_fft_extent(input[0], kernel[0], context, "rows")?;
    let cols = checked_fft_extent(input[1], kernel[1], context, "cols")?;
    shape_2d([rows, cols], context, "FFT buffer")
}

pub(super) fn checked_fft_shape_3d(
    input: [usize; 3],
    kernel: [usize; 3],
    context: &str,
) -> Result<Shape3d> {
    require_nonzero_3d(input, context, "input")?;
    require_nonzero_3d(kernel, context, "kernel")?;
    let depth = checked_fft_extent(input[0], kernel[0], context, "depth")?;
    let rows = checked_fft_extent(input[1], kernel[1], context, "rows")?;
    let cols = checked_fft_extent(input[2], kernel[2], context, "cols")?;
    shape_3d([depth, rows, cols], context, "FFT buffer")
}

pub(super) fn edge_source_index(padded_index: usize, radius: usize, extent: usize) -> usize {
    debug_assert!(extent > 0);
    padded_index.saturating_sub(radius).min(extent - 1)
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
        slice_len,
        len,
    })
}

fn require_nonzero_2d(shape: [usize; 2], context: &str, role: &str) -> Result<()> {
    if shape.iter().all(|&extent| extent > 0) {
        return Ok(());
    }
    Err(anyhow!(
        "{context}: {role} dimensions must be non-zero, got [{}, {}]",
        shape[0],
        shape[1]
    ))
}

fn require_nonzero_3d(shape: [usize; 3], context: &str, role: &str) -> Result<()> {
    if shape.iter().all(|&extent| extent > 0) {
        return Ok(());
    }
    Err(anyhow!(
        "{context}: {role} dimensions must be non-zero, got [{}, {}, {}]",
        shape[0],
        shape[1],
        shape[2]
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        checked_edge_shape_2d, checked_edge_shape_3d, checked_fft_shape_2d, checked_fft_shape_3d,
        edge_source_index,
    };

    #[test]
    fn checked_fft_shape_2d_uses_linear_convolution_extent() {
        let shape = checked_fft_shape_2d([5, 6], [3, 4], "fft2").unwrap();

        assert_eq!((shape.rows, shape.cols, shape.len), (8, 16, 128));
    }

    #[test]
    fn checked_fft_shape_3d_tracks_slice_and_total_len() {
        let shape = checked_fft_shape_3d([5, 6, 7], [3, 4, 5], "fft3").unwrap();

        assert_eq!(
            (
                shape.depth,
                shape.rows,
                shape.cols,
                shape.slice_len,
                shape.len
            ),
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
        let error = checked_fft_shape_2d([oversized_extent, 8], [1, 1], "fft2").unwrap_err();

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
        let error = checked_fft_shape_3d([oversized_depth, 2, 2], [1, 1, 1], "fft3").unwrap_err();

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
