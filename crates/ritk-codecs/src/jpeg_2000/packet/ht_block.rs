//! Decode T.814 cleanup, significance propagation and magnitude refinement
//! segments, then restore reversible coefficients in the RITK wavelet domain.
use anyhow::{ensure, Result};
pub(super) fn decode(
    data: &[u8],
    width: usize,
    height: usize,
    planes: u32,
    missing: u32,
    passes: u32,
    cleanup: u32,
) -> Result<Vec<i64>> {
    ensure!(
        (1..=62).contains(&planes) && missing < planes && (1..=3).contains(&passes),
        "invalid HT code-block bit-planes or passes"
    );
    let mut sign_magnitude = vec![0u64; width * height];
    let refinement = u32::try_from(data.len())?
        .checked_sub(cleanup)
        .ok_or_else(|| anyhow::anyhow!("short HT cleanup segment"))?;
    ensure!(
        htj2k::coding::decode_codeblock64(
            data,
            &mut sign_magnitude,
            missing,
            passes,
            cleanup,
            refinement,
            u32::try_from(width)?,
            u32::try_from(height)?,
            u32::try_from(width)?,
            false
        ),
        "invalid HT code-block entropy data"
    );
    let shift = 63 - planes;
    Ok(sign_magnitude
        .into_iter()
        .map(|word| {
            let magnitude = ((word & i64::MAX as u64) >> shift) as i64;
            if word >> 63 != 0 {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect())
}
