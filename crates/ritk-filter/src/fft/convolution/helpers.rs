use num_complex::Complex;
use apollo_fft::FftPlan1D;
use apollo_fft::domain::metadata::shape::Shape1D;

// ── ZST FFT direction strategy ──────────────────────────────────────────────

/// Trait for FFT transform direction.
///
/// Each implementation is a zero-sized type so that the compiler monomorphises
/// the FFT functions with the direction fully inlined and the match branch
/// eliminated — zero runtime overhead versus a hand-written variant.
pub trait FftDirection: Default {
    /// Create an FFT plan for the given transform length.
    fn plan(len: usize) -> FftPlan1D<f32>;

    /// Process the slice in-place.
    fn process(plan: &FftPlan1D<f32>, slice: &mut [Complex<f32>]);
}

/// Forward FFT direction: spatial → frequency domain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ForwardFft;

impl FftDirection for ForwardFft {
    #[inline]
    fn plan(len: usize) -> FftPlan1D<f32> {
        FftPlan1D::<f32>::new(Shape1D::new(len).expect("FFT length must be > 0"))
    }

    #[inline]
    fn process(plan: &FftPlan1D<f32>, slice: &mut [Complex<f32>]) {
        plan.forward_complex_slice_inplace(slice);
    }
}

/// Inverse FFT direction: frequency → spatial domain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InverseFft;

impl FftDirection for InverseFft {
    #[inline]
    fn plan(len: usize) -> FftPlan1D<f32> {
        FftPlan1D::<f32>::new(Shape1D::new(len).expect("FFT length must be > 0"))
    }

    #[inline]
    fn process(plan: &FftPlan1D<f32>, slice: &mut [Complex<f32>]) {
        plan.inverse_complex_slice_unnorm_inplace(slice);
    }
}

// ── Generic FFT functions ────────────────────────────────────────────────────

/// Dispatch separable N-D FFT based on dimensionality.
///
/// For `D = 2`, delegates to `fft2d`; for `D = 3`, delegates to `fft3d`.
/// Panics for any other `D` value.
pub fn fft_nd<const D: usize, Dir: FftDirection>(
    buf: &mut [Complex<f32>],
    dims: &[usize; D],
) {
    match D {
        2 => fft2d::<Dir>(buf, dims[0], dims[1]),
        3 => fft3d::<Dir>(buf, dims[0], dims[1], dims[2]),
        _ => panic!("fft_nd: only D=2 and D=3 are supported, got D={D}"),
    }
}

/// In-place separable 2-D FFT (or IFFT) on a row-major buffer of shape
/// `[rows, cols]`.
///
/// Pass 1: 1-D transform along each row (transform length = `cols`).
/// Pass 2: 1-D transform along each column via a scratch column buffer
/// (transform length = `rows`).
pub fn fft2d<Dir: FftDirection>(
    buf: &mut [Complex<f32>],
    rows: usize,
    cols: usize,
) {
    let row_fft = Dir::plan(cols);
    let col_fft = Dir::plan(rows);

    // Row-wise pass.
    for r in 0..rows {
        Dir::process(&row_fft, &mut buf[r * cols..(r + 1) * cols]);
    }

    // Column-wise pass via scratch buffer.
    let mut col_buf = vec![Complex::new(0.0_f32, 0.0); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = buf[r * cols + c];
        }
        Dir::process(&col_fft, &mut col_buf);
        for r in 0..rows {
            buf[r * cols + c] = col_buf[r];
        }
    }
}

/// In-place separable 3-D FFT (or IFFT) on a row-major buffer of shape
/// `[depth, rows, cols]`.
///
/// Pass 1: 1-D transform along each row (transform length = `cols`).
/// Pass 2: 1-D transform along each column (transform length = `rows`).
/// Pass 3: 1-D transform along the depth axis (transform length = `depth`).
pub fn fft3d<Dir: FftDirection>(
    buf: &mut [Complex<f32>],
    depth: usize,
    rows: usize,
    cols: usize,
) {
    let row_fft = Dir::plan(cols);
    let col_fft = Dir::plan(rows);
    let depth_fft = Dir::plan(depth);

    let slice = rows * cols;

    // Row-wise pass: for each (depth, row), transform along cols.
    for d in 0..depth {
        for r in 0..rows {
            Dir::process(&row_fft, &mut buf[d * slice + r * cols..d * slice + (r + 1) * cols]);
        }
    }

    // Column-wise pass: for each (depth, col), transform along rows.
    let mut col_buf = vec![Complex::new(0.0_f32, 0.0); rows];
    for d in 0..depth {
        for c in 0..cols {
            for r in 0..rows {
                col_buf[r] = buf[d * slice + r * cols + c];
            }
            Dir::process(&col_fft, &mut col_buf);
            for r in 0..rows {
                buf[d * slice + r * cols + c] = col_buf[r];
            }
        }
    }

    // Depth-wise pass: for each (row, col), transform along depth.
    let mut depth_buf = vec![Complex::new(0.0_f32, 0.0); depth];
    for r in 0..rows {
        for c in 0..cols {
            for d in 0..depth {
                depth_buf[d] = buf[d * slice + r * cols + c];
            }
            Dir::process(&depth_fft, &mut depth_buf);
            for d in 0..depth {
                buf[d * slice + r * cols + c] = depth_buf[d];
            }
        }
    }
}
