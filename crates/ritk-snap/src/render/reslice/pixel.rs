//! Continuous-pixel sampling and patient-plane projection.

use super::sampling::sample_volume;
use super::{ResliceError, ReslicePlane};
use crate::LoadedVolume;

mod interval;
use interval::Interval;

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

impl ReslicePlane {
    // The forward map uses two rounded multiplies and two rounded additions
    // per patient component. The standard model bounds their accumulated
    // relative error by gamma_4 = 4u / (1 - 4u); four minimum subnormals cover
    // underflow in those operations (Higham, Accuracy and Stability of
    // Numerical Algorithms, 2nd ed., §2.2). For inverse coefficients A, the
    // forward error gives m <= b + αm; solving this componentwise 2×2 bound
    // makes the enclosure local to p and rejects points outside a small plane.
    // Exact zero coefficients stay exact during interval propagation, so
    // normal displacement cannot widen unrelated pixel axes.
    pub(super) fn patient_pixel_enclosure(self, patient: [f64; 3]) -> Option<[Interval; 2]> {
        let basis = PixelProjectionBasis::new(self.horizontal_step, self.vertical_step)?;
        let origin = self.origin();
        let centered_patient = std::array::from_fn(|axis| {
            Interval::point(patient[axis]).subtract(Interval::point(origin[axis]))
        });
        let [Some(delta_x), Some(delta_y), Some(delta_z)] = centered_patient else {
            return None;
        };
        let center = basis.project([delta_x, delta_y, delta_z])?;
        let coefficients = [basis.column, basis.row];
        let unit_roundoff = f64::EPSILON * 0.5;
        let relative_bound = (4.0 * unit_roundoff / (1.0 - 4.0 * unit_roundoff)).next_up();
        let underflow_bound = 4.0 * f64::from_bits(1);
        let horizontal = self.horizontal_step.map(f64::abs);
        let vertical = self.vertical_step.map(f64::abs);

        let growth = std::array::from_fn(|pixel_axis| {
            std::array::from_fn(|source_axis| {
                coefficients[pixel_axis]
                    .into_iter()
                    .enumerate()
                    .try_fold(Interval::point(0.0), |sum, (patient_axis, coefficient)| {
                        let step = if source_axis == 0 {
                            horizontal[patient_axis]
                        } else {
                            vertical[patient_axis]
                        };
                        sum.add(
                            Interval::point(coefficient.magnitude_bound())
                                .multiply(Interval::point(relative_bound))?
                                .multiply(Interval::point(step))?,
                        )
                    })
                    .map(Interval::upper)
            })
        });
        let [[Some(a00), Some(a01)], [Some(a10), Some(a11)]] = growth else {
            return None;
        };
        let one = Interval::point(1.0);
        let one_minus_a00 = one.subtract(Interval::point(a00))?;
        let one_minus_a11 = one.subtract(Interval::point(a11))?;
        if one_minus_a00.lower() <= 0.0 || one_minus_a11.lower() <= 0.0 {
            return None;
        }
        let determinant = one_minus_a00
            .multiply(one_minus_a11)?
            .subtract(Interval::point(a01).multiply(Interval::point(a10))?)?;
        if determinant.lower() <= 0.0 {
            return None;
        }

        let offset = std::array::from_fn(|pixel_axis| {
            coefficients[pixel_axis]
                .into_iter()
                .enumerate()
                .try_fold(Interval::point(0.0), |sum, (patient_axis, coefficient)| {
                    let coordinate_rounding = Interval::point(origin[patient_axis].abs())
                        .multiply(Interval::point(relative_bound))?
                        .add(Interval::point(underflow_bound))?;
                    sum.add(
                        Interval::point(coefficient.magnitude_bound())
                            .multiply(coordinate_rounding)?,
                    )
                })
                .map(Interval::upper)
        });
        let [Some(column_offset), Some(row_offset)] = offset else {
            return None;
        };
        let column_bound =
            Interval::point(center[0].magnitude_bound()).add(Interval::point(column_offset))?;
        let row_bound =
            Interval::point(center[1].magnitude_bound()).add(Interval::point(row_offset))?;
        let maximum_pixel = [
            one_minus_a11
                .multiply(column_bound)?
                .add(Interval::point(a01).multiply(row_bound)?)?
                .divide(determinant)?
                .upper(),
            Interval::point(a10)
                .multiply(column_bound)?
                .add(one_minus_a00.multiply(row_bound)?)?
                .divide(determinant)?
                .upper(),
        ];

        let uncertain_delta = std::array::from_fn(|axis| {
            let magnitude = Interval::point(origin[axis].abs())
                .add(
                    Interval::point(horizontal[axis])
                        .multiply(Interval::point(maximum_pixel[0]))?,
                )?
                .add(
                    Interval::point(vertical[axis]).multiply(Interval::point(maximum_pixel[1]))?,
                )?;
            let radius = magnitude
                .multiply(Interval::point(relative_bound))?
                .add(Interval::point(underflow_bound))?
                .upper();
            Interval::symmetric(patient[axis], radius)?.subtract(Interval::point(origin[axis]))
        });
        let [Some(delta_x), Some(delta_y), Some(delta_z)] = uncertain_delta else {
            return None;
        };
        basis.project([delta_x, delta_y, delta_z])
    }

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
        let patient = self.patient_at_pixel(pixel)?.coordinates();
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
}

#[derive(Clone, Copy)]
struct PixelProjectionBasis {
    column: [Interval; 3],
    row: [Interval; 3],
}

impl PixelProjectionBasis {
    fn new(horizontal: [f64; 3], vertical: [f64; 3]) -> Option<Self> {
        let horizontal = horizontal.map(Interval::point);
        let vertical = vertical.map(Interval::point);
        let horizontal_length = interval_norm(horizontal)?;
        let horizontal_unit = horizontal.map(|component| component.divide(horizontal_length));
        let [Some(horizontal_x), Some(horizontal_y), Some(horizontal_z)] = horizontal_unit else {
            return None;
        };
        let horizontal_unit = [horizontal_x, horizontal_y, horizontal_z];
        let vertical_projection = interval_dot(vertical, horizontal_unit)?;
        let vertical_residual = std::array::from_fn(|axis| {
            vertical[axis].subtract(horizontal_unit[axis].multiply(vertical_projection)?)
        });
        let [Some(residual_x), Some(residual_y), Some(residual_z)] = vertical_residual else {
            return None;
        };
        let vertical_residual = [residual_x, residual_y, residual_z];
        let vertical_length = interval_norm(vertical_residual)?;
        let vertical_unit = vertical_residual.map(|component| component.divide(vertical_length));
        let [Some(vertical_x), Some(vertical_y), Some(vertical_z)] = vertical_unit else {
            return None;
        };
        let vertical_unit = [vertical_x, vertical_y, vertical_z];
        let row = vertical_unit.map(|component| component.divide(vertical_length));
        let [Some(row_x), Some(row_y), Some(row_z)] = row else {
            return None;
        };
        let row = [row_x, row_y, row_z];
        let column = std::array::from_fn(|axis| {
            horizontal_unit[axis]
                .subtract(vertical_projection.multiply(row[axis])?)?
                .divide(horizontal_length)
        });
        let [Some(column_x), Some(column_y), Some(column_z)] = column else {
            return None;
        };
        Some(Self {
            column: [column_x, column_y, column_z],
            row,
        })
    }

    fn project(self, delta: [Interval; 3]) -> Option<[Interval; 2]> {
        Some([
            interval_dot(delta, self.column)?,
            interval_dot(delta, self.row)?,
        ])
    }
}

fn interval_dot(first: [Interval; 3], second: [Interval; 3]) -> Option<Interval> {
    first
        .into_iter()
        .zip(second)
        .try_fold(Interval::point(0.0), |sum, (left, right)| {
            sum.add(left.multiply(right)?)
        })
}

fn interval_norm(vector: [Interval; 3]) -> Option<Interval> {
    let scale = vector
        .into_iter()
        .map(Interval::magnitude_bound)
        .fold(0.0, f64::max);
    if scale == 0.0 {
        return Some(Interval::point(0.0));
    }
    let scale = Interval::point(scale);
    let normalized = vector.map(|component| component.divide(scale));
    let [Some(x), Some(y), Some(z)] = normalized else {
        return None;
    };
    [x, y, z]
        .into_iter()
        .try_fold(Interval::point(0.0), |sum, component| {
            sum.add(component.multiply(component)?)
        })?
        .square_root()?
        .multiply(scale)
}
