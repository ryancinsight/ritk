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

use crate::native_support::map_flat_image;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::tensor::Backend;
use ritk_image::{native::Image as NativeImage, Image};
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
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals_vec, dims) = extract_vec_infallible(image);
        let vals = &vals_vec;

        let factor = self.suv_factor();

        let out_vals: Vec<f32> = vals.iter().map(|&v| (v as f64 * factor) as f32).collect();

        Ok(rebuild(out_vals, dims, image))
    }

    /// Apply the SUVbw conversion to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &NativeImage<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<NativeImage<f32, B, 3>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let factor = self.suv_factor();
        map_flat_image(image, backend, move |values, _| {
            values
                .iter()
                .map(|&value| (f64::from(value) * factor) as f32)
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use coeus_core::SequentialBackend;
    use ritk_image::native::Image as NativeImage;
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};

    type B = coeus_core::SequentialBackend;

    fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<f32, B, 3> {
        use ritk_image::tensor::Tensor;
        let tensor = Tensor::<f32, B>::from_slice(shape, &vals);
        Image::new(
            tensor,
            Point::new([0.0_f64, 0.0, 0.0]),
            Spacing::new([1.0_f64, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn voxels(img: &Image<f32, B, 3>) -> Vec<f32> {
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

    #[test]
    fn native_suv_matches_the_tensor_backed_values_and_metadata() {
        let filter = SuvBodyWeightImageFilter::new(75.0, 350e6);
        let values = vec![0.0, 10_000.0, 20_000.0];
        let legacy = filter
            .apply(&make_image(values.clone(), [1, 1, 3]))
            .expect("tensor-backed SUV succeeds");
        let native = NativeImage::from_flat_on(
            values,
            [1, 1, 3],
            Point::new([1.0, 2.0, 3.0]),
            Spacing::new([0.5, 1.0, 2.0]),
            Direction::identity(),
            &SequentialBackend,
        )
        .expect("invariant: valid native image");
        let output = filter
            .apply_native(&native, &SequentialBackend)
            .expect("native SUV succeeds");

        assert_eq!(
            output.data_slice().expect("invariant: contiguous storage"),
            voxels(&legacy)
        );
        assert_eq!(output.origin(), native.origin());
        assert_eq!(output.spacing(), native.spacing());
        assert_eq!(output.direction(), native.direction());
    }
}
