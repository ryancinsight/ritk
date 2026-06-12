//! D = 1: Medial point extraction.
//!
//! For each maximal connected foreground run, retain only the midpoint.

/// For each maximal connected foreground run [a, b], retain only the
/// midpoint ⌊(a + b) / 2⌋. This preserves one point per connected
/// component (topology) and selects the medial position.
pub(super) fn endpoint_extract(flat: &[f32], nx: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; nx];
    let mut i = 0;
    while i < nx {
        if flat[i] > 0.5 {
            let start = i;
            while i < nx && flat[i] > 0.5 {
                i += 1;
            }
            // Run is [start, i-1] inclusive.
            let mid = (start + (i - 1)) / 2;
            output[mid] = 1.0;
        } else {
            i += 1;
        }
    }
    output
}
