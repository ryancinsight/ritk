use super::{ImageReadBudget, ImageReadBudgetError, ImageReadResource};

#[test]
fn default_budget_matches_the_documented_resource_ceilings() {
    assert_eq!(ImageReadBudget::DEFAULT.max_encoded_bytes(), 1_073_741_824);
    assert_eq!(ImageReadBudget::DEFAULT.max_decoded_bytes(), 1_073_741_824);
    assert_eq!(ImageReadBudget::DEFAULT.max_series_volumes(), 65_536);
}

#[test]
fn budget_construction_rejects_each_zero_ceiling() {
    assert_eq!(
        ImageReadBudget::new(0, 1, 1),
        Err(ImageReadBudgetError::ZeroLimit {
            resource: ImageReadResource::EncodedBytes
        })
    );
    assert_eq!(
        ImageReadBudget::new(1, 0, 1),
        Err(ImageReadBudgetError::ZeroLimit {
            resource: ImageReadResource::DecodedBytes
        })
    );
    assert_eq!(
        ImageReadBudget::new(1, 1, 0),
        Err(ImageReadBudgetError::ZeroLimit {
            resource: ImageReadResource::SeriesVolumes
        })
    );
}

#[test]
fn budget_accepts_limits_and_reports_each_resource_overrun() {
    let budget = ImageReadBudget::new(32, 64, 3).expect("all ceilings are positive");
    assert_eq!(budget.check(ImageReadResource::EncodedBytes, 32), Ok(()));
    assert_eq!(budget.check(ImageReadResource::DecodedBytes, 64), Ok(()));
    assert_eq!(budget.check(ImageReadResource::SeriesVolumes, 3), Ok(()));
    assert_eq!(
        budget.check(ImageReadResource::EncodedBytes, 33),
        Err(ImageReadBudgetError::Exceeded {
            resource: ImageReadResource::EncodedBytes,
            actual: 33,
            maximum: 32,
        })
    );
    assert_eq!(
        budget.check(ImageReadResource::DecodedBytes, 65),
        Err(ImageReadBudgetError::Exceeded {
            resource: ImageReadResource::DecodedBytes,
            actual: 65,
            maximum: 64,
        })
    );
    assert_eq!(
        budget.check(ImageReadResource::SeriesVolumes, 4),
        Err(ImageReadBudgetError::Exceeded {
            resource: ImageReadResource::SeriesVolumes,
            actual: 4,
            maximum: 3,
        })
    );
}

#[test]
fn length_check_applies_the_selected_ceiling() {
    let budget = ImageReadBudget::new(8, 16, 2).expect("all ceilings are positive");
    assert_eq!(budget.check_len(ImageReadResource::EncodedBytes, 8), Ok(()));
    assert_eq!(
        budget.check_len(ImageReadResource::DecodedBytes, 17),
        Err(ImageReadBudgetError::Exceeded {
            resource: ImageReadResource::DecodedBytes,
            actual: 17,
            maximum: 16,
        })
    );
}

#[test]
fn scaled_check_rejects_a_product_beyond_usize_under_the_widest_ceiling() {
    let budget = ImageReadBudget::new(1, u64::MAX, 1).expect("all ceilings are positive");
    assert_eq!(
        budget.check_scaled(ImageReadResource::DecodedBytes, usize::MAX, 2),
        Err(ImageReadBudgetError::Exceeded {
            resource: ImageReadResource::DecodedBytes,
            actual: u64::MAX,
            maximum: u64::MAX,
        })
    );
}

#[test]
fn scaled_check_applies_the_ceiling_to_a_product_that_fits_usize() {
    let admitting = ImageReadBudget::new(1, 16, 1).expect("all ceilings are positive");
    assert_eq!(
        admitting.check_scaled(ImageReadResource::DecodedBytes, 4, 4),
        Ok(())
    );
    let rejecting = ImageReadBudget::new(1, 15, 1).expect("all ceilings are positive");
    assert_eq!(
        rejecting.check_scaled(ImageReadResource::DecodedBytes, 4, 4),
        Err(ImageReadBudgetError::Exceeded {
            resource: ImageReadResource::DecodedBytes,
            actual: 16,
            maximum: 15,
        })
    );
}
