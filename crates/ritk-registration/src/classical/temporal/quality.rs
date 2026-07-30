//! Residual diagnostics over the valid interpolated signal overlap.

use leto::Array1;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ResidualMetrics {
    pub(crate) overlap_samples: usize,
    pub(crate) rms: f64,
    pub(crate) max_abs: f64,
}

pub(crate) fn aligned_residual_metrics(
    reference: &Array1<f64>,
    moving: &Array1<f64>,
    shift_frames: f64,
) -> ResidualMetrics {
    let mut overlap_samples = 0;
    let mut squared_residual_sum = 0.0;
    let mut max_abs = 0.0_f64;

    for (reference_index, &reference_value) in reference.iter().enumerate() {
        let moving_coordinate = reference_index as f64 + shift_frames;
        let Some(moving_value) = interpolate(moving, moving_coordinate) else {
            continue;
        };

        let residual = reference_value - moving_value;
        squared_residual_sum += residual * residual;
        max_abs = max_abs.max(residual.abs());
        overlap_samples += 1;
    }

    debug_assert!(
        overlap_samples > 0,
        "validated search bounds retain a non-empty aligned overlap"
    );
    ResidualMetrics {
        overlap_samples,
        rms: (squared_residual_sum / overlap_samples as f64).sqrt(),
        max_abs,
    }
}

pub(crate) fn interpolate(signal: &Array1<f64>, coordinate: f64) -> Option<f64> {
    if coordinate < 0.0 || coordinate > signal.size().saturating_sub(1) as f64 {
        return None;
    }

    let lower = coordinate.floor() as usize;
    let fraction = coordinate - lower as f64;
    let lower_value = *signal.get([lower]).ok()?;
    if fraction == 0.0 {
        return Some(lower_value);
    }

    let upper_value = *signal.get([lower.checked_add(1)?]).ok()?;
    Some((upper_value - lower_value).mul_add(fraction, lower_value))
}
