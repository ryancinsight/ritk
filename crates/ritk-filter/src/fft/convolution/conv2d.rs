use crate::fft::convolution::helpers::{fft2d, ForwardFft, InverseFft};
use crate::fft::convolution::padding::{
    checked_edge_shape_2d, checked_fft_shape_2d, edge_source_index,
};
use anyhow::{anyhow, Result};
use eunomia::Complex;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};
use std::marker::PhantomData;

// ── FftConvolutionFilter ───────────────────────────────────────────────────────

/// FFT-based 2-D convolution filter ("same" output convention).
///
/// Stores the raw kernel values at construction time. At `apply` time the
/// padded FFTs of both image and kernel are computed, multiplied in the
/// frequency domain, and the "same"-sized output is cropped from the IFFT
/// result. Any image size is accepted regardless of the kernel size.
///
/// # Complexity
///
/// O(N log N) where N = pad_r · pad_c.
///
/// # Output
///
/// The output has the same spatial shape as the input and preserves origin,
/// spacing, and direction metadata.
pub struct FftConvolutionFilter<B: Backend> {
    kernel_vals: Vec<f32>,
    kernel_rows: usize,
    kernel_cols: usize,
    _phantom: PhantomData<fn() -> B>,
}

impl<B: Backend> FftConvolutionFilter<B> {
    /// Construct from a 2-D kernel image.
    pub fn new(kernel: &Image<B, 2>) -> Result<Self> {
        let [kr, kc] = kernel.shape();
        if kr == 0 || kc == 0 {
            return Err(anyhow!(
                "FftConvolutionFilter: kernel dimensions must be non-zero, got [{kr}, {kc}]"
            ));
        }
        let (k_vals, _) = extract_vec(kernel)?;
        Ok(Self {
            kernel_vals: k_vals,
            kernel_rows: kr,
            kernel_cols: kc,
            _phantom: PhantomData::<fn() -> B>,
        })
    }

    /// Convolve `image` with the stored kernel ("same" convention).
    ///
    /// # Mathematical contract
    ///
    /// For an odd-sized kernel with `δ[kr/2, kc/2] = 1` and all other entries
    /// zero, `apply(image)` reproduces `image` within floating-point precision.
    pub fn apply(&self, image: &Image<B, 2>) -> Result<Image<B, 2>> {
        let [h, w] = image.shape();
        let (vals, dims) = extract_vec(image)?;

        let kr = self.kernel_rows;
        let kc = self.kernel_cols;
        let off_r = kr / 2;
        let off_c = kc / 2;

        // ZeroFluxNeumann boundary (matching ITK): edge-replicate-pad by the
        // kernel radius, convolve the larger image with zero padding, then crop
        // the central window so each original-border pixel sees a full
        // edge-clamped kernel footprint.
        let padded_shape = checked_edge_shape_2d([h, w], [off_r, off_c], "FftConvolutionFilter")?;
        let (ph, pw) = (padded_shape.rows, padded_shape.cols);
        let mut padded = vec![0.0_f32; padded_shape.len];
        for r in 0..ph {
            let sr = edge_source_index(r, off_r, h);
            for c in 0..pw {
                let sc = edge_source_index(c, off_c, w);
                padded[r * pw + c] = vals[sr * w + sc];
            }
        }

        let conv = self.convolve_same(&padded, [ph, pw])?;

        let mut out = vec![0.0_f32; h * w];
        for r in 0..h {
            for c in 0..w {
                out[r * w + c] = conv[(r + off_r) * pw + (c + off_c)];
            }
        }

        Ok(rebuild(out, dims, image))
    }

    /// FFT linear convolution with "same" output (zero padding, no boundary
    /// extension). `dims` is the input shape `[h, w]`.
    fn convolve_same(&self, vals: &[f32], dims: [usize; 2]) -> Result<Vec<f32>> {
        let [h, w] = dims;
        let kr = self.kernel_rows;
        let kc = self.kernel_cols;

        // Padding must be >= h + kr − 1 to suppress circular aliasing.
        let fft_shape = checked_fft_shape_2d(dims, [kr, kc], "FftConvolutionFilter")?;
        let (pad_r, pad_c, pad_n) = (fft_shape.rows, fft_shape.cols, fft_shape.len);

        // Zero-padded image: placed at top-left (origin).
        let mut img_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for r in 0..h {
            for c in 0..w {
                img_buf[r * pad_c + c] = Complex::new(vals[r * w + c], 0.0);
            }
        }

        // Zero-padded kernel: placed at top-left (origin).
        let mut ker_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for r in 0..kr {
            for c in 0..kc {
                ker_buf[r * pad_c + c] = Complex::new(self.kernel_vals[r * kc + c], 0.0);
            }
        }

        fft2d::<ForwardFft>(&mut img_buf, pad_r, pad_c);
        fft2d::<ForwardFft>(&mut ker_buf, pad_r, pad_c);

        // Point-wise complex multiply: img_buf[i] *= ker_buf[i].
        // (a + bi)(c + di) = (ac − bd) + (ad + bc)i
        for i in 0..pad_n {
            let a = img_buf[i];
            let b = ker_buf[i];
            img_buf[i] = Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
        }

        fft2d::<InverseFft>(&mut img_buf, pad_r, pad_c);

        // Normalize by 1/pad_n and extract "same" window at (⌊kr/2⌋, ⌊kc/2⌋).
        let scale = 1.0_f32 / pad_n as f32;
        let off_r = kr / 2;
        let off_c = kc / 2;

        let mut out = vec![0.0_f32; h * w];
        for r in 0..h {
            for c in 0..w {
                out[r * w + c] = img_buf[(r + off_r) * pad_c + (c + off_c)].re * scale;
            }
        }

        Ok(out)
    }
}
