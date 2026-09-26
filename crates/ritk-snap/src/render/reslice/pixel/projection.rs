//! Patient-space projection onto a validated reslice plane.

use super::super::ReslicePlane;
use super::interval::Interval;
use super::PixelMappingError;
use crate::geometry::PatientPointMm;

const UNIT_ROUNDOFF: f64 = f64::EPSILON * 0.5;
const FORWARD_OPERATION_COUNT: f64 = 4.0;
const MIN_SUBNORMAL: f64 = f64::from_bits(1);
const MAX_PIXEL_ERROR: f64 = 0.5;

/// A patient point projected into continuous output pixels and plane distance.
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
    ///
    /// Positive distance points along the horizontal cross vertical normal.
    #[must_use]
    pub const fn distance_mm(self) -> f64 {
        self.distance_mm
    }
}

impl ReslicePlane {
    /// Project patient millimetres into this plane's physical pixel basis.
    ///
    /// Off-plane points retain their signed distance. The inverse enclosure must
    /// extend less than half a pixel from the nominal coordinate on every axis;
    /// adjacent pixel centres are one coordinate unit apart, so a wider enclosure
    /// cannot resolve a subpixel location. A nominal coordinate just beyond an
    /// edge is clamped only when its enclosure contains that edge; an enclosure
    /// wholly outside the pixel extent is rejected. A nominal result outside its
    /// enclosure is unresolved and is not returned.
    ///
    /// The enclosure accounts for the two rounded multiplies and two rounded
    /// additions in the forward pixel mapping. Its relative bound is
    /// `gamma_4 = 4u / (1 - 4u)` with `u = EPSILON / 2`; four least subnormals
    /// bound underflow in those operations. The componentwise inverse bound
    /// solves `m <= b + A m` for the nonnegative 2×2 error matrix `A`. The
    /// floating-point model follows Higham, *Accuracy and Stability of
    /// Numerical Algorithms*, 2nd ed., [§2.2 and eq. (2.8)][chapter 2] for rounding
    /// and gradual underflow, and [Lemma 3.1, p. 63][chapter 3] for `gamma_n`.
    /// The componentwise 2×2 bound is derived here.
    ///
    /// [chapter 2]: https://epubs.siam.org/doi/10.1137/1.9780898718027.ch2
    /// [chapter 3]: https://epubs.siam.org/doi/10.1137/1.9780898718027.ch3
    ///
    /// # Errors
    ///
    /// Returns [`PixelMappingError::InvalidPatientPoint`] for non-finite input,
    /// [`PixelMappingError::ProjectionOverflow`] when finite arithmetic
    /// overflows, [`PixelMappingError::ProjectionUnresolved`] when the result
    /// cannot be confirmed by its error enclosure, or
    /// [`PixelMappingError::OutOfBounds`] when the projected pixel is outside
    /// the output plane.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use ritk_snap::render::{ResliceInterpolation, ReslicePlane};
    /// use ritk_snap::LoadedVolume;
    ///
    /// let volume = LoadedVolume {
    ///     data: Arc::new(vec![0.0; 8]),
    ///     shape: [2, 2, 2],
    ///     channels: 1,
    ///     spacing: [1.0; 3],
    ///     origin: [0.0; 3],
    ///     direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ///     metadata: None,
    ///     source: None,
    ///     modality: None,
    ///     patient_name: None,
    ///     patient_id: None,
    ///     study_date: None,
    ///     series_description: None,
    ///     series_time: None,
    ///     patient_weight_kg: None,
    ///     injected_dose_bq: None,
    ///     radionuclide_half_life_s: None,
    ///     radiopharmaceutical_start_time: None,
    ///     decay_correction: None,
    /// };
    /// let plane = ReslicePlane::try_new(
    ///     &volume,
    ///     [0.0; 3],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0; 3],
    ///     [2, 2],
    ///     1,
    ///     ResliceInterpolation::Nearest,
    /// )
    /// .expect("invariant: the 2x2 plane fits the source volume");
    /// let projection = plane
    ///     .project_patient([0.5, 0.25, 1.5])
    ///     .expect("invariant: the finite point has a resolvable projection");
    /// assert_eq!(projection.pixel(), [0.5, 0.25]);
    /// assert_eq!(projection.distance_mm(), 1.5);
    /// ```
    pub fn project_patient(
        self,
        patient: [f64; 3],
    ) -> Result<PatientPlaneProjection, PixelMappingError> {
        let patient = PatientPointMm::try_from(patient)
            .map_err(|source| PixelMappingError::InvalidPatientPoint { source })?
            .coordinates();
        let delta = std::array::from_fn(|axis| patient[axis] - self.origin[axis]);
        if !delta.into_iter().all(f64::is_finite) {
            return Err(PixelMappingError::ProjectionOverflow {
                coordinate: patient,
            });
        }

        let basis = ProjectionBasis::new(self.horizontal_step, self.vertical_step).ok_or(
            PixelMappingError::ProjectionUnresolved {
                coordinate: patient,
            },
        )?;
        let pixel = basis
            .project(delta)
            .ok_or(PixelMappingError::ProjectionOverflow {
                coordinate: patient,
            })?;
        let normal = cross(basis.horizontal_unit, basis.vertical_unit);
        let normal_length = vector_norm(normal);
        if !normal_length.is_finite() || normal_length == 0.0 {
            return Err(PixelMappingError::ProjectionUnresolved {
                coordinate: patient,
            });
        }
        let normal = normal.map(|component| component / normal_length);
        let distance_mm =
            scaled_dot(delta, normal).ok_or(PixelMappingError::ProjectionOverflow {
                coordinate: patient,
            })?;
        let enclosure = self.patient_pixel_enclosure(patient).ok_or(
            PixelMappingError::ProjectionUnresolved {
                coordinate: patient,
            },
        )?;
        if !is_resolved(pixel, enclosure) {
            return Err(PixelMappingError::ProjectionUnresolved {
                coordinate: patient,
            });
        }
        let limits = self.maximum_pixel_coordinate();
        let pixel = [
            clamp_or_reject(pixel[0], enclosure[0], limits[0], pixel, self.dimensions)?,
            clamp_or_reject(pixel[1], enclosure[1], limits[1], pixel, self.dimensions)?,
        ];
        Ok(PatientPlaneProjection { pixel, distance_mm })
    }

    pub(in crate::render::reslice) fn patient_pixel_enclosure(
        self,
        patient: [f64; 3],
    ) -> Option<[Interval; 2]> {
        let basis = PixelProjectionBasis::new(self.horizontal_step, self.vertical_step)?;
        let origin = self.origin;
        let centered_patient = std::array::from_fn(|axis| {
            Interval::point(patient[axis]).subtract(Interval::point(origin[axis]))
        });
        let [Some(delta_x), Some(delta_y), Some(delta_z)] = centered_patient else {
            return None;
        };
        let center = basis.project([delta_x, delta_y, delta_z])?;
        let coefficients = [basis.column, basis.row];
        let relative_bound = (FORWARD_OPERATION_COUNT * UNIT_ROUNDOFF
            / (1.0 - FORWARD_OPERATION_COUNT * UNIT_ROUNDOFF))
            .next_up();
        let underflow_bound = FORWARD_OPERATION_COUNT * MIN_SUBNORMAL;
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
}

fn is_resolved(pixel: [f64; 2], enclosure: [Interval; 2]) -> bool {
    pixel.into_iter().zip(enclosure).all(|(coordinate, bound)| {
        bound.contains(coordinate)
            && coordinate - bound.lower() < MAX_PIXEL_ERROR
            && bound.upper() - coordinate < MAX_PIXEL_ERROR
    })
}

fn clamp_or_reject(
    coordinate: f64,
    enclosure: Interval,
    limit: f64,
    projected: [f64; 2],
    dimensions: [usize; 2],
) -> Result<f64, PixelMappingError> {
    if coordinate < 0.0 {
        if enclosure.contains(0.0) {
            return Ok(0.0);
        }
    } else if coordinate > limit {
        if enclosure.contains(limit) {
            return Ok(limit);
        }
    } else if enclosure.upper() < 0.0 || enclosure.lower() > limit {
        return Err(PixelMappingError::OutOfBounds {
            coordinate: projected,
            dimensions,
        });
    } else {
        return Ok(coordinate);
    }
    Err(PixelMappingError::OutOfBounds {
        coordinate: projected,
        dimensions,
    })
}

#[derive(Clone, Copy)]
struct ProjectionBasis {
    horizontal_unit: [f64; 3],
    vertical_unit: [f64; 3],
    horizontal_length: f64,
    vertical_length: f64,
    vertical_projection: f64,
}

impl ProjectionBasis {
    fn new(horizontal: [f64; 3], vertical: [f64; 3]) -> Option<Self> {
        let horizontal_length = vector_norm(horizontal);
        if !horizontal_length.is_finite() || horizontal_length == 0.0 {
            return None;
        }
        let horizontal_unit = horizontal.map(|component| component / horizontal_length);
        let vertical_projection = scaled_dot(vertical, horizontal_unit)?;
        let vertical_residual = std::array::from_fn(|axis| {
            (-horizontal_unit[axis]).mul_add(vertical_projection, vertical[axis])
        });
        if !vertical_residual.into_iter().all(f64::is_finite) {
            return None;
        }
        let vertical_length = vector_norm(vertical_residual);
        if !vertical_length.is_finite() || vertical_length == 0.0 {
            return None;
        }
        let vertical_unit = vertical_residual.map(|component| component / vertical_length);
        Some(Self {
            horizontal_unit,
            vertical_unit,
            horizontal_length,
            vertical_length,
            vertical_projection,
        })
    }

    fn project(self, delta: [f64; 3]) -> Option<[f64; 2]> {
        let row = scaled_dot(delta, self.vertical_unit)? / self.vertical_length;
        let horizontal = scaled_dot(delta, self.horizontal_unit)?;
        let column = (-self.vertical_projection).mul_add(row, horizontal) / self.horizontal_length;
        (row.is_finite() && column.is_finite()).then_some([column, row])
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

fn vector_norm(vector: [f64; 3]) -> f64 {
    vector[0].hypot(vector[1]).hypot(vector[2])
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1].mul_add(right[2], -left[2] * right[1]),
        left[2].mul_add(right[0], -left[0] * right[2]),
        left[0].mul_add(right[1], -left[1] * right[0]),
    ]
}

fn scaled_dot(left: [f64; 3], right: [f64; 3]) -> Option<f64> {
    let [left_x, left_y, left_z] = left;
    let [right_x, right_y, right_z] = right;
    let products = [left_x * right_x, left_y * right_y, left_z * right_z];
    if !products.into_iter().all(f64::is_finite) {
        return None;
    }
    let scale = products.into_iter().map(f64::abs).fold(0.0, f64::max);
    if scale == 0.0 {
        return Some(0.0);
    }
    let [product_x, product_y, product_z] = products;
    let normalized = product_x / scale + product_y / scale + product_z / scale;
    let result = normalized * scale;
    result.is_finite().then_some(result)
}
