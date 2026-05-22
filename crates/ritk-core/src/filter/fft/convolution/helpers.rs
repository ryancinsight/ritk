use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

// ── Private helpers ────────────────────────────────────────────────────────────

/// FFT transform direction.
pub enum FftDir {
    Forward,
    Inverse,
}

/// In-place separable 2-D FFT (or IFFT) on a row-major buffer of shape
/// `[rows, cols]`.
///
/// Pass 1: 1-D transform along each row (transform length = `cols`).
/// Pass 2: 1-D transform along each column via a scratch column buffer
///         (transform length = `rows`).
///
/// `rustfft`'s `process` method performs the transform in-place and allocates
/// scratch space internally.
pub fn fft2d(
    buf: &mut [Complex<f32>],
    rows: usize,
    cols: usize,
    planner: &mut FftPlanner<f32>,
    dir: FftDir,
) {
    let row_fft = match dir {
        FftDir::Forward => planner.plan_fft_forward(cols),
        FftDir::Inverse => planner.plan_fft_inverse(cols),
    };
    let col_fft = match dir {
        FftDir::Forward => planner.plan_fft_forward(rows),
        FftDir::Inverse => planner.plan_fft_inverse(rows),
    };

    // Row-wise pass.
    for r in 0..rows {
        row_fft.process(&mut buf[r * cols..(r + 1) * cols]);
    }

    // Column-wise pass via scratch buffer.
    let mut col_buf = vec![Complex::new(0.0_f32, 0.0); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = buf[r * cols + c];
        }
        col_fft.process(&mut col_buf);
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
///
/// `rustfft`'s `process` method performs the transform in-place and allocates
/// scratch space internally.
pub fn fft3d(
    buf: &mut [Complex<f32>],
    depth: usize,
    rows: usize,
    cols: usize,
    planner: &mut FftPlanner<f32>,
    dir: FftDir,
) {
    let row_fft = match dir {
        FftDir::Forward => planner.plan_fft_forward(cols),
        FftDir::Inverse => planner.plan_fft_inverse(cols),
    };
    let col_fft = match dir {
        FftDir::Forward => planner.plan_fft_forward(rows),
        FftDir::Inverse => planner.plan_fft_inverse(rows),
    };
    let depth_fft = match dir {
        FftDir::Forward => planner.plan_fft_forward(depth),
        FftDir::Inverse => planner.plan_fft_inverse(depth),
    };

    let slice = rows * cols;

    // Row-wise pass: for each (depth, row), transform along cols.
    for d in 0..depth {
        for r in 0..rows {
            row_fft.process(&mut buf[d * slice + r * cols..d * slice + (r + 1) * cols]);
        }
    }

    // Column-wise pass: for each (depth, col), transform along rows.
    let mut col_buf = vec![Complex::new(0.0_f32, 0.0); rows];
    for d in 0..depth {
        for c in 0..cols {
            for r in 0..rows {
                col_buf[r] = buf[d * slice + r * cols + c];
            }
            col_fft.process(&mut col_buf);
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
            depth_fft.process(&mut depth_buf);
            for d in 0..depth {
                buf[d * slice + r * cols + c] = depth_buf[d];
            }
        }
    }
}
