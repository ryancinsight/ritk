//! Precision-dependent JPEG-LS sample and NEAR limits.

const MAX_HEADER_NEAR: u32 = u8::MAX as u32;

pub(super) fn maximum_sample_for_precision(precision: u32) -> u32 {
    debug_assert!(
        (1..=16).contains(&precision),
        "JPEG-LS precision must fit the supported sample representation"
    );
    (1u32 << precision) - 1
}

pub(super) fn maximum_near_for_precision(precision: u32) -> u32 {
    MAX_HEADER_NEAR.min(maximum_sample_for_precision(precision) / 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_limit_tracks_sample_range_and_header_width() {
        assert_eq!(maximum_near_for_precision(2), 1);
        assert_eq!(maximum_near_for_precision(8), 127);
        assert_eq!(maximum_near_for_precision(9), 255);
        assert_eq!(maximum_near_for_precision(16), 255);
    }
}
