pub mod histogram;
pub mod image_comparison;
pub mod image_statistics;
pub mod information;
pub mod jacobian;
pub mod label_overlap;
pub mod label_shape_extended;
pub mod label_statistics;
pub mod noise_estimation;
pub mod normalization;
pub mod position_extrema;
pub mod value_indices;
pub use histogram::{histogram, histogram_from_slice, Histogram};
pub use image_comparison::{
    dice_coefficient, dice_coefficient_native, hausdorff_distance, hausdorff_distance_native,
    mean_surface_distance, mean_surface_distance_native, pearson_correlation, psnr, psnr_native,
    similarity_index, ssim, ssim_native,
};
pub use image_statistics::{compute_statistics, masked_statistics, ImageStatistics};
pub use information::{
    conditional_mutual_information, interaction_information, joint_entropy, joint_entropy_n,
    marginal_entropy, multivariate_variation_of_information, mutual_information,
    mutual_information_mattes, normalized_mutual_information, symmetric_uncertainty,
    total_correlation, variation_of_information,
};
pub use jacobian::{analyze_jacobian, jacobian_determinant, JacobianStats};
pub use label_overlap::{
    label_overlap_measures, label_overlap_measures_from_slices, LabelOverlapMeasures,
};
pub use label_shape_extended::{
    compute_label_shape_statistics_extended, compute_label_shape_statistics_extended_from_slices,
    LabelShapeStatisticsExtended,
};
pub use label_statistics::{
    compute_label_intensity_statistics, compute_label_intensity_statistics_from_slices,
    LabelIntensityStatistics,
};
pub use noise_estimation::{
    estimate_noise_mad, estimate_noise_mad_masked, estimate_noise_mad_native,
};
pub use normalization::{
    HistogramMatcher, IntensityRange, MinMaxNormalizer, MriContrast, NyulUdupaNormalizer,
    WhiteStripeConfig, WhiteStripeNormalizer, WhiteStripeResult, ZScoreNormalizer,
};
pub use position_extrema::{maximum_position, minimum_position};
pub use value_indices::{value_indices, ValueIndices};

/// Sort a mutable slice of `f32` values using total ordering (NaN sorted last).
#[inline]
pub(crate) fn sort_floats(values: &mut [f32]) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Binary-mask foreground threshold: voxels with mask value strictly above
/// this threshold are treated as foreground; those at or below are background.
pub(crate) const FOREGROUND_THRESHOLD: f32 = 0.5;

#[cfg(test)]
mod tests_label_overlap;

#[cfg(test)]
mod tests_label_shape_extended;
