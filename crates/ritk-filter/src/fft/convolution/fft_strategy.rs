use eunomia::Complex;

// ── ZST FFT direction strategy ──────────────────────────────────────────────

/// Trait for FFT transform direction.
///
/// Each implementation is a zero-sized type so that the compiler monomorphises
/// the FFT functions with the direction fully inlined and the match branch
/// eliminated — zero runtime overhead versus a hand-written variant.
pub trait FftDirection: Default {
    /// Process a 1-D slice in-place.
    fn process(slice: &mut [Complex<f32>]);
}

/// Forward FFT direction: spatial → frequency domain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ForwardFft;

impl FftDirection for ForwardFft {
    #[inline]
    fn process(slice: &mut [Complex<f32>]) {
        apollo_fft::application::execution::kernel::fft_forward(slice);
    }
}

/// Inverse FFT direction: frequency → spatial domain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InverseFft;

impl FftDirection for InverseFft {
    #[inline]
    fn process(slice: &mut [Complex<f32>]) {
        // Keep inverse unnormalized so outer callers can apply one global 1/N
        // normalization after all axis passes.
        apollo_fft::application::execution::kernel::fft_inverse_unnorm(slice);
    }
}

// ── Generic FFT functions ────────────────────────────────────────────────────

/// Dispatch separable N-D FFT based on dimensionality.
///
/// For `D = 2`, delegates to `fft2d`; for `D = 3`, delegates to `fft3d`.
/// Panics for any other `D` value.
pub fn fft_nd<const D: usize, Dir: FftDirection>(buf: &mut [Complex<f32>], dims: &[usize; D]) {
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
pub fn fft2d<Dir: FftDirection>(buf: &mut [Complex<f32>], rows: usize, cols: usize) {
    // Row-wise pass.
    for r in 0..rows {
        Dir::process(&mut buf[r * cols..(r + 1) * cols]);
    }

    // Column-wise pass via scratch buffer.
    let mut col_buf = vec![Complex::new(0.0_f32, 0.0); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = buf[r * cols + c];
        }
        Dir::process(&mut col_buf);
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
pub fn fft3d<Dir: FftDirection>(buf: &mut [Complex<f32>], depth: usize, rows: usize, cols: usize) {
    let slice = rows * cols;

    // Row-wise pass: for each (depth, row), transform along cols.
    for d in 0..depth {
        for r in 0..rows {
            Dir::process(&mut buf[d * slice + r * cols..d * slice + (r + 1) * cols]);
        }
    }

    // Column-wise pass: for each (depth, col), transform along rows.
    let mut col_buf = vec![Complex::new(0.0_f32, 0.0); rows];
    for d in 0..depth {
        for c in 0..cols {
            for r in 0..rows {
                col_buf[r] = buf[d * slice + r * cols + c];
            }
            Dir::process(&mut col_buf);
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
            Dir::process(&mut depth_buf);
            for d in 0..depth {
                buf[d * slice + r * cols + c] = depth_buf[d];
            }
        }
    }
}
