//! Continuous-pixel sampling and patient-plane projection.

use super::sampling::{add_scaled, sample_volume};
use super::{ResliceError, ReslicePlane};
use crate::LoadedVolume;

/// A scalar sample and physical mapping for one continuous reslice pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResliceSample {
    pixel: [f64; 2],
    patient: [f64; 3],
    voxel: [f64; 3],
    nearest_voxel: [usize; 3],
    value: f32,
}

impl ResliceSample {
    /// Return the continuous output coordinate in `[column, row]` order.
    #[must_use]
    pub const fn pixel(self) -> [f64; 2] {
        self.pixel
    }

    /// Return the corresponding patient-space coordinate in millimetres.
    #[must_use]
    pub const fn patient(self) -> [f64; 3] {
        self.patient
    }

    /// Return the corresponding continuous `[depth, row, column]` coordinate.
    #[must_use]
    pub const fn voxel(self) -> [f64; 3] {
        self.voxel
    }

    /// Return the nearest in-bounds `[depth, row, column]` source voxel.
    #[must_use]
    pub const fn nearest_voxel(self) -> [usize; 3] {
        self.nearest_voxel
    }

    /// Return the source scalar sampled with the plane's interpolation policy.
    #[must_use]
    pub const fn value(self) -> f32 {
        self.value
    }
}

/// Coordinates of a patient-space point projected into a reslice plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatientPlaneProjection {
    pixel: [f64; 2],
    distance_mm: f64,
}

impl PatientPlaneProjection {
    /// Return the projected output coordinate in `[column, row]` order.
    #[must_use]
    pub const fn pixel(self) -> [f64; 2] {
        self.pixel
    }

    /// Return the signed point-to-plane distance in millimetres.
    #[must_use]
    pub const fn distance_mm(self) -> f64 {
        self.distance_mm
    }
}

impl ReslicePlane {
    /// Map and sample a continuous output-pixel coordinate on this plane.
    ///
    /// The coordinate is `[column, row]`, where integer coordinates identify
    /// output-pixel centres. The scalar is sampled at the first through-plane
    /// position using the plane's interpolation policy.
    ///
    /// # Errors
    ///
    /// Returns a coordinate error when the pixel is non-finite or outside the
    /// output dimensions and preserves source-shape/geometry validation.
    pub fn sample_pixel(
        self,
        volume: &LoadedVolume,
        pixel: [f64; 2],
    ) -> Result<ResliceSample, ResliceError> {
        let transform = self.source_transform(volume)?;
        if !pixel.into_iter().all(f64::is_finite) {
            return Err(ResliceError::InvalidPixelCoordinate { coordinate: pixel });
        }
        let maximum = [
            self.dimensions[0].saturating_sub(1) as f64,
            self.dimensions[1].saturating_sub(1) as f64,
        ];
        if pixel
            .into_iter()
            .zip(maximum)
            .any(|(coordinate, limit)| coordinate < 0.0 || coordinate > limit)
        {
            return Err(ResliceError::PixelOutOfBounds {
                coordinate: pixel,
                dimensions: self.dimensions,
            });
        }

        let patient = add_scaled(
            add_scaled(self.origin, self.horizontal_step, pixel[0]),
            self.vertical_step,
            pixel[1],
        );
        let voxel = transform.patient_to_voxel(patient);
        let value = sample_volume(volume, voxel, self.interpolation)?;
        let nearest_voxel = voxel.map(f64::round).map(|coordinate| {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the sampled voxel coordinate is finite and inside the source extent"
            )]
            {
                coordinate as usize
            }
        });
        Ok(ResliceSample {
            pixel,
            patient,
            voxel,
            nearest_voxel,
            value,
        })
    }

    /// Project a patient-space point onto the plane's physical basis.
    ///
    /// The returned distance is positive on the normal side defined by the
    /// horizontal × vertical basis. Projection does not clip to the output
    /// dimensions, allowing callers to apply their own viewport bounds.
    #[must_use]
    pub fn project_patient(self, patient: [f64; 3]) -> Option<PatientPlaneProjection> {
        if !patient.into_iter().all(f64::is_finite) {
            return None;
        }
        let delta = subtract(patient, self.origin);
        let horizontal = self.horizontal_step;
        let vertical = self.vertical_step;
        let hh = dot(horizontal, horizontal);
        let hv = dot(horizontal, vertical);
        let vv = dot(vertical, vertical);
        let determinant = hh * vv - hv * hv;
        if !determinant.is_finite() || determinant <= 0.0 {
            return None;
        }
        let dh = dot(delta, horizontal);
        let dv = dot(delta, vertical);
        let pixel = [
            (dh * vv - dv * hv) / determinant,
            (dv * hh - dh * hv) / determinant,
        ];
        if !pixel.into_iter().all(f64::is_finite) {
            return None;
        }
        let residual = subtract(
            delta,
            add_scaled(
                add_scaled([0.0; 3], horizontal, pixel[0]),
                vertical,
                pixel[1],
            ),
        );
        let normal = cross(horizontal, vertical);
        let normal_length = norm(normal);
        if !normal_length.is_finite() || normal_length == 0.0 {
            return None;
        }
        let signed_distance = dot(residual, normal) / normal_length;
        signed_distance
            .is_finite()
            .then_some(PatientPlaneProjection {
                pixel,
                distance_mm: signed_distance,
            })
    }
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(vector: [f64; 3]) -> f64 {
    vector[0].hypot(vector[1]).hypot(vector[2])
}
