//! Scalar sample conversion from the stored representation to compute intensities.

use super::{NiftiDatatype, NiftiHeader, NiftiLane};
use anyhow::{bail, Result};
use ritk_image_io::{CalibrationError, LinearCalibration};

impl NiftiHeader {
    /// A zero slope disables both scaling fields under NIfTI's data scaling rule.
    /// Otherwise the coefficients must represent a finite linear transform.
    pub(crate) fn linear_calibration(&self) -> Result<Option<LinearCalibration>, CalibrationError> {
        if self.scl_slope == 0.0 {
            return Ok(None);
        }
        LinearCalibration::new(self.scl_slope, self.scl_inter).map(Some)
    }

    pub(crate) fn decode_intensities(
        &self,
        bytes: &[u8],
        calibration: Option<LinearCalibration>,
    ) -> Result<Vec<f32>> {
        let values = match self.datatype {
            NiftiDatatype::Uint8 => self.convert_lanes::<u8, 1>(bytes, calibration, f32::from),
            NiftiDatatype::Int16 => self.convert_lanes::<i16, 2>(bytes, calibration, f32::from),
            NiftiDatatype::Int32 => self.convert_lanes::<i32, 4>(bytes, calibration, |v| v as f32),
            NiftiDatatype::Uint32 => self.convert_lanes::<u32, 4>(bytes, calibration, |v| v as f32),
            NiftiDatatype::Float32 => self.convert_lanes::<f32, 4>(bytes, calibration, |v| v),
            datatype => bail!(
                "NIfTI image convenience reader does not support stored datatype {datatype:?}"
            ),
        };
        Ok(values)
    }

    fn convert_lanes<T: NiftiLane<N> + Into<f64>, const N: usize>(
        &self,
        bytes: &[u8],
        calibration: Option<LinearCalibration>,
        unscaled: impl Fn(T) -> f32,
    ) -> Vec<f32> {
        // Header and payload validation has already established exactly N bytes per lane.
        let (lanes, remainder) = bytes.as_chunks::<N>();
        debug_assert!(remainder.is_empty());
        let stored = lanes.iter().map(|lane| self.read_lane::<T, N>(*lane));
        if let Some(scale) = calibration {
            // All supported integer and binary32 lanes are exact in binary64.
            // NIfTI-2 coefficients are binary64. Narrow only after applying
            // both coefficients, so a large integer minus an offset retains
            // its low bits. The fused operation rounds the affine sum once.
            return stored
                .map(|v| scale.slope().mul_add(v.into(), scale.intercept()) as f32)
                .collect();
        }
        // In particular, a binary32 identity decode retains signed zero and NaN bits.
        stored.map(unscaled).collect()
    }
}
