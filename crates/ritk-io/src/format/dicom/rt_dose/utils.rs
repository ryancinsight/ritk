//! Private helper utilities for RT Dose I/O.

/// Parse a `\`-separated DICOM Decimal String into a fixed-size `f64` array.
///
/// Returns `None` when fewer than `N` parseable values are present.
pub(super) fn parse_ds_backslash<const N: usize>(s: &str) -> Option<[f64; N]> {
    let parts: Vec<f64> = s
        .trim()
        .split('\\')
        .filter_map(|p| p.trim().parse::<f64>().ok())
        .collect();
    if parts.len() >= N {
        let mut arr = [0.0_f64; N];
        arr.copy_from_slice(&parts[..N]);
        Some(arr)
    } else {
        None
    }
}
