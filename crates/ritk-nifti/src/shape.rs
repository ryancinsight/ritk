use anyhow::{anyhow, Result};

pub(crate) fn checked_voxel_count(nx: usize, ny: usize, nz: usize) -> Result<usize> {
    nx.checked_mul(ny)
        .and_then(|xy| xy.checked_mul(nz))
        .ok_or_else(|| anyhow!("NIfTI voxel count overflows usize: nx={nx}, ny={ny}, nz={nz}"))
}

#[cfg(test)]
mod tests {
    use super::checked_voxel_count;

    #[test]
    fn checked_voxel_count_multiplies_dimensions() {
        assert_eq!(
            checked_voxel_count(4, 3, 2).expect("small dimensions must multiply"),
            24
        );
    }

    #[test]
    fn checked_voxel_count_rejects_overflow() {
        let err = checked_voxel_count(usize::MAX, 2, 1)
            .expect_err("overflowing NIfTI dimensions must be rejected");

        assert!(
            err.to_string().contains("overflows usize"),
            "error must name overflow invariant: {err}"
        );
    }
}
