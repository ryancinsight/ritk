//! Standardized Uptake Value (SUV) filters.
//!
//! # Mathematical Specification
//!
//! SUV Body Weight (SUVbw) normalizes tissue radioactivity concentration by the
//! injected dose and the patient's body weight.
//!
//! `SUVbw = (ActivityConcentration_BqML) * (PatientWeight_kg * 1000.0) / InjectedDose_Bq`
//!
//! This assumes the image `ActivityConcentration` and the `InjectedDose` are
//! calibrated to the same reference time (e.g. `DecayCorrection = START` or `ADMIN`).
//!
//! ## Invariants
//!
//! - Spatial metadata (shape, origin, spacing, direction) is preserved exactly.
//! - Transforms voxels from Bq/mL to unitless SUV.

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Convert PET activity concentration (Bq/mL) to SUV Body Weight (SUVbw).
///
/// # Example
///
/// ```no_run
/// # use ritk_filter::SuvBodyWeightImageFilter;
/// // patient_weight = 75.0 kg, total_dose = 350,000,000 Bq
/// let filter = SuvBodyWeightImageFilter::new(75.0, 350e6);
/// ```
#[derive(Debug, Clone)]
pub struct SuvBodyWeightImageFilter {
    /// Patient weight in kg (DICOM 0010,1030).
    pub patient_weight_kg: f64,
    /// Radionuclide total dose in Bq (DICOM 0018,1074).
    pub total_dose_bq: f64,
}

impl SuvBodyWeightImageFilter {
    /// Create a new SUVbw filter.
    ///
    /// # Arguments
    /// * `patient_weight_kg` - Patient weight in kilograms.
    /// * `total_dose_bq` - Total injected dose in Becquerels.
    pub fn new(patient_weight_kg: f64, total_dose_bq: f64) -> Self {
        Self {
            patient_weight_kg,
            total_dose_bq,
        }
    }

    /// Calculate the multiplicative factor to convert Bq/mL to SUVbw.
    #[inline]
    pub fn suv_factor(&self) -> f64 {
        // SUVbw = (Bq/mL) * (kg * 1000 g/kg) / Bq
        if self.total_dose_bq <= 0.0 {
            return 0.0;
        }
        (self.patient_weight_kg * 1000.0) / self.total_dose_bq
    }

    /// Apply the filter to a 3-D image.
    ///
    /// Returns a new `Image` with identical spatial metadata and
    /// voxel values transformed to SUVbw.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec_infallible(image);
        let vals = &vals_vec;

        let factor = self.suv_factor();

        let out_vals: Vec<f32> = vals.iter().map(|&v| (v as f64 * factor) as f32).collect();

        Ok(rebuild(out_vals, dims, image))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        use burn::tensor::{Shape, Tensor, TensorData};
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0_f64, 0.0, 0.0]),
            Spacing::new([1.0_f64, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn voxels(img: &Image<B, 3>) -> Vec<f32> {
        img.data_slice().into_owned()
    }

    #[test]
    fn test_suv_factor_calculation() {
        // 75 kg patient, 350 MBq dose
        let filter = SuvBodyWeightImageFilter::new(75.0, 350e6);
        let factor = filter.suv_factor();
        // 75000 / 350000000 = 0.0002142857...
        let expected = 75000.0 / 350_000_000.0;
        assert!((factor - expected).abs() < 1e-9);
    }

    #[test]
    fn test_suv_apply() {
        let filter = SuvBodyWeightImageFilter::new(75.0, 350e6);
        let factor = filter.suv_factor() as f32;

        // 10000 Bq/mL should become ~2.14 SUV
        let img = make_image(vec![0.0, 10000.0, 20000.0], [1, 1, 3]);
        let out = filter.apply(&img).unwrap();
        let v = voxels(&out);

        assert!((v[0] - 0.0).abs() < 1e-5);
        assert!((v[1] - 10000.0 * factor).abs() < 1e-5);
        assert!((v[2] - 20000.0 * factor).abs() < 1e-5);
    }

    #[test]
    fn test_suv_zero_dose_safety() {
        let filter = SuvBodyWeightImageFilter::new(75.0, 0.0);
        assert_eq!(filter.suv_factor(), 0.0);
    }
}
