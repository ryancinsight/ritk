//! Flat-buffer native trilinear interpolation.
//!
//! Zero-copy path for callers that already hold contiguous host buffers and
//! do not need the `ritk_image::native::Image` wrapper overhead. The function
//! here is a SSOT for the flat-buffer trilinear kernel; the Image-backed sister
//! in [`crate::interpolation::trilinear_interpolation`] delegates to the same
//! math via `coeus_ops::linear_interpolation`.
//!
//! Buffer layout contract (matches `ritk_image` row-major / C order):
//! - `image`: `[b, c, d, h, w]` row-major
//! - `grid`:  `[b, 3, out_d, out_h, out_w]` channel 0=z, 1=y, 2=x
//! - output:  `[b, c, out_d, out_h, out_w]`

use eunomia::{CastFrom, FloatElement};

/// Trilinear image sampling on a flat voxel buffer.
///
/// Grid channels are `(z, y, x)` in voxel space. Out-of-bounds coordinates
/// are clamped to the border (replicate padding).
///
/// # Panics
/// Panics if buffer lengths don't match the declared dimensions.
#[allow(clippy::too_many_arguments)]
pub fn trilinear_interpolation<T>(
    image: &[T],
    b: usize,
    c: usize,
    d: usize,
    h: usize,
    w: usize,
    grid: &[T],
    out_d: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<T>
where
    T: Copy + FloatElement,
    usize: CastFrom<T>,
{
    assert_eq!(
        image.len(),
        b * c * d * h * w,
        "trilinear_interpolation: image buffer length {} != {}*{}*{}*{}*{}",
        image.len(),
        b,
        c,
        d,
        h,
        w
    );
    assert_eq!(
        grid.len(),
        b * 3 * out_d * out_h * out_w,
        "trilinear_interpolation: grid buffer length {} != {}*3*{}*{}*{}",
        grid.len(),
        b,
        out_d,
        out_h,
        out_w
    );

    let out_sp = out_d * out_h * out_w;
    let g_ch = out_sp;
    let g_b = 3 * out_sp;
    let i_ch = d * h * w;
    let i_b = c * i_ch;
    let one = T::ONE;

    let mut out = vec![T::ZERO; b * c * out_sp];

    for bi in 0..b {
        let gb = bi * g_b;
        for p in 0..out_sp {
            let gz = grid[gb + p];
            let gy = grid[gb + g_ch + p];
            let gx = grid[gb + 2 * g_ch + p];

            let (z0, z1, fz) = split(gz, d);
            let (y0, y1, fy) = split(gy, h);
            let (x0, x1, fx) = split(gx, w);

            let w000 = (one - fz) * (one - fy) * (one - fx);
            let w001 = (one - fz) * (one - fy) * fx;
            let w010 = (one - fz) * fy * (one - fx);
            let w011 = (one - fz) * fy * fx;
            let w100 = fz * (one - fy) * (one - fx);
            let w101 = fz * (one - fy) * fx;
            let w110 = fz * fy * (one - fx);
            let w111 = fz * fy * fx;

            for ci in 0..c {
                let base = bi * i_b + ci * i_ch;
                let v = at(image, base, z0, y0, x0, h, w) * w000
                    + at(image, base, z0, y0, x1, h, w) * w001
                    + at(image, base, z0, y1, x0, h, w) * w010
                    + at(image, base, z0, y1, x1, h, w) * w011
                    + at(image, base, z1, y0, x0, h, w) * w100
                    + at(image, base, z1, y0, x1, h, w) * w101
                    + at(image, base, z1, y1, x0, h, w) * w110
                    + at(image, base, z1, y1, x1, h, w) * w111;
                out[(bi * c + ci) * out_sp + p] = v;
            }
        }
    }
    out
}

#[inline]
fn split<T>(coord: T, size: usize) -> (usize, usize, T)
where
    T: Copy + FloatElement,
    usize: CastFrom<T>,
{
    let max = T::from_f64(size.saturating_sub(1) as f64);
    let c = coord.max(T::ZERO).min(max);
    let fl = c.floor();
    let i0 = usize::cast_from(fl).min(size.saturating_sub(1));
    let i1 = (i0 + 1).min(size.saturating_sub(1));
    (i0, i1, c - fl)
}

#[inline]
fn at<T: Copy>(buf: &[T], base: usize, zi: usize, yi: usize, xi: usize, h: usize, w: usize) -> T {
    buf[base + zi * h * w + yi * w + xi]
}

#[cfg(test)]
mod tests {
    use super::trilinear_interpolation;
    use eunomia::{CastFrom, FloatElement};

    fn assert_sampling_contract<T>()
    where
        T: Copy + core::fmt::Debug + PartialEq + FloatElement,
        usize: CastFrom<T>,
    {
        let image = (0..8)
            .map(|value| T::from_f64(f64::from(value)))
            .collect::<Vec<_>>();

        let center = [T::from_f64(0.5); 3];
        let center_sample = trilinear_interpolation(&image, 1, 1, 2, 2, 2, &center, 1, 1, 1);
        assert_eq!(center_sample, vec![T::from_f64(3.5)]);

        let upper = [T::from_f64(4.0); 3];
        let upper_sample = trilinear_interpolation(&image, 1, 1, 2, 2, 2, &upper, 1, 1, 1);
        assert_eq!(upper_sample, vec![T::from_f64(7.0)]);
    }

    #[test]
    fn flat_buffer_sampling_preserves_scalar_contract() {
        assert_sampling_contract::<f32>();
        assert_sampling_contract::<f64>();
    }
}
