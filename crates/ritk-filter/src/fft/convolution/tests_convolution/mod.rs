//! Tests for `FftConvolutionFilter` and `FftNormalizedCorrelationFilter`.
//!
//! Each test is derived from a closed-form mathematical specification so that
//! the acceptance criterion is analytically verifiable, not empirically tuned.

mod conv_2d;
mod conv_3d_ncc_3d;
mod ncc_2d;
