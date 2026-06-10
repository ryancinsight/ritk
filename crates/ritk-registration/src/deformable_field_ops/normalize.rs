//! Force normalization for diffeomorphic registration.

/// Normalize two force fields by their maximum magnitude and a gradient step.
///
/// Scales `u1` and `u2` so their max component equals `gradient_step`,
/// preventing instability from large gradient magnitudes.
///
/// # Invariants
/// - If all components of `u1` are zero (below 1e-10), `u1` is left unchanged.
/// - If all components of `u2` are zero (below 1e-10), `u2` is left unchanged.
/// - Computation is performed in `f64` to avoid intermediate precision loss;
///   the scaling factor is cast to `f32` only for the final multiply.
#[inline]
pub(crate) fn normalize_forces_into(
    u1z: &mut [f32],
    u1y: &mut [f32],
    u1x: &mut [f32],
    u2z: &mut [f32],
    u2y: &mut [f32],
    u2x: &mut [f32],
    gradient_step: f64,
) {
    let max_u1 = u1z
        .iter()
        .chain(u1y.iter())
        .chain(u1x.iter())
        .map(|&v| (v as f64).abs())
        .fold(0.0_f64, f64::max);
    if max_u1 > 1e-10 {
        let s = (gradient_step / max_u1) as f32;
        u1z.iter_mut().for_each(|v| *v *= s);
        u1y.iter_mut().for_each(|v| *v *= s);
        u1x.iter_mut().for_each(|v| *v *= s);
    }

    let max_u2 = u2z
        .iter()
        .chain(u2y.iter())
        .chain(u2x.iter())
        .map(|&v| (v as f64).abs())
        .fold(0.0_f64, f64::max);
    if max_u2 > 1e-10 {
        let s = (gradient_step / max_u2) as f32;
        u2z.iter_mut().for_each(|v| *v *= s);
        u2y.iter_mut().for_each(|v| *v *= s);
        u2x.iter_mut().for_each(|v| *v *= s);
    }
}
