//! Private helper utilities for RT Structure Set parsing.

/// Parse a `\`-separated DS contour-data string into 3-D point triples.
///
/// # Invariant
/// Output length = `floor(parseable_token_count / 3)`.
/// Non-numeric tokens are silently discarded. The result always contains
/// complete `[X, Y, Z]` triples; partial trailing values are dropped.
pub(super) fn parse_contour_data(s: &str) -> Vec<[f64; 3]> {
    let vals: Vec<f64> = s
        .split('\\')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    vals.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Parse a `\`-separated IS color string into an `[R, G, B]` u8 triple.
///
/// # Invariant
/// Returns `Some([r, g, b])` iff the string contains at least 3 parseable u8 values.
pub(super) fn parse_color(s: &str) -> Option<[u8; 3]> {
    let v: Vec<u8> = s
        .split('\\')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    if v.len() >= 3 {
        Some([v[0], v[1], v[2]])
    } else {
        None
    }
}
