//! Private helper utilities for RT Dose I/O.

use anyhow::{bail, Context, Result};

/// Parse a `\`-separated DICOM Decimal String into a fixed-size `f64` array.
///
/// Returns `Ok` only when exactly `N` numeric components are present.
pub(super) fn parse_ds_backslash<const N: usize>(s: &str, field: &str) -> Result<[f64; N]> {
    let tokens: Vec<&str> = s.trim().split('\\').collect();
    if tokens.len() != N {
        bail!(
            "{} must contain exactly {} DS components, got {}",
            field,
            N,
            tokens.len()
        );
    }

    let mut arr = [0.0_f64; N];
    for (index, token) in tokens.into_iter().enumerate() {
        arr[index] = token
            .trim()
            .parse::<f64>()
            .with_context(|| format!("Invalid {} component {}: '{}'", field, index, token))?;
    }
    Ok(arr)
}

/// Parse `GridFrameOffsetVector` into exactly one offset per frame.
pub(super) fn parse_frame_offsets(s: &str, n_frames: usize) -> Result<Vec<f64>> {
    let tokens: Vec<&str> = s.trim().split('\\').collect();
    if tokens.len() != n_frames {
        bail!(
            "GridFrameOffsetVector must contain exactly {} offsets, got {}",
            n_frames,
            tokens.len()
        );
    }

    tokens
        .into_iter()
        .enumerate()
        .map(|(index, token)| {
            token.trim().parse::<f64>().with_context(|| {
                format!(
                    "Invalid GridFrameOffsetVector component {}: '{}'",
                    index, token
                )
            })
        })
        .collect()
}
