//! Private helper utilities for RT Structure Set parsing.

use anyhow::{bail, Context, Result};

/// Parse a `\`-separated DS contour-data string into 3-D point triples.
///
/// # Invariant
/// Returns `Ok(points)` iff every token parses as `f64` and the component
/// count is an exact multiple of 3.
pub(super) fn parse_contour_data(s: &str) -> Result<Vec<[f64; 3]>> {
    if s.trim().is_empty() {
        bail!("ContourData must contain at least one point");
    }

    let component_count = s.split('\\').count();
    if !component_count.is_multiple_of(3) {
        bail!(
            "ContourData component count must be divisible by 3, got {}",
            component_count
        );
    }

    let mut points = Vec::with_capacity(component_count / 3);
    let mut point = [0.0f64; 3];
    for (index, token) in s.split('\\').enumerate() {
        point[index % 3] = token
            .trim()
            .parse::<f64>()
            .with_context(|| format!("Invalid ContourData component {}: '{}'", index, token))?;
        if index % 3 == 2 {
            points.push(point);
        }
    }
    Ok(points)
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
