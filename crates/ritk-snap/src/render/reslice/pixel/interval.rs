//! Outward-rounded intervals for patient-to-pixel projection.

// The exact product of two binary64 significands can have 106 bits, so its
// least nonzero residual bit may be 105 places below the product's scale.
// Below 2^-968 that residual can round to zero even when it is nonzero.
const RESIDUAL_UNDERFLOW_LIMIT: f64 = f64::MIN_POSITIVE * 4.0 / f64::EPSILON;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::render::reslice) struct Interval {
    lower: f64,
    upper: f64,
}

impl Interval {
    pub(super) const fn point(value: f64) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }

    pub(super) fn symmetric(center: f64, radius: f64) -> Option<Self> {
        let lower = directed_add(center, -radius, true)?;
        let upper = directed_add(center, radius, false)?;
        (lower <= upper).then_some(Self { lower, upper })
    }

    pub(in crate::render::reslice) const fn lower(self) -> f64 {
        self.lower
    }

    pub(in crate::render::reslice) const fn upper(self) -> f64 {
        self.upper
    }

    pub(in crate::render::reslice) const fn contains(self, value: f64) -> bool {
        self.lower <= value && value <= self.upper
    }

    pub(super) fn magnitude_bound(self) -> f64 {
        self.lower.abs().max(self.upper.abs())
    }

    pub(super) fn add(self, other: Self) -> Option<Self> {
        if other.is_zero() {
            return Some(self);
        }
        if self.is_zero() {
            return Some(other);
        }
        Some(Self {
            lower: directed_add(self.lower, other.lower, true)?,
            upper: directed_add(self.upper, other.upper, false)?,
        })
    }

    pub(super) fn subtract(self, other: Self) -> Option<Self> {
        if other.is_zero() {
            return Some(self);
        }
        self.add(other.negate())
    }

    pub(super) fn multiply(self, other: Self) -> Option<Self> {
        if self.is_zero() || other.is_zero() {
            return Some(Self::point(0.0));
        }
        let products = [
            (self.lower, other.lower),
            (self.lower, other.upper),
            (self.upper, other.lower),
            (self.upper, other.upper),
        ];
        let [first, second, third, fourth] = products;
        let lower = directed_multiply(first.0, first.1, true)?
            .min(directed_multiply(second.0, second.1, true)?)
            .min(directed_multiply(third.0, third.1, true)?)
            .min(directed_multiply(fourth.0, fourth.1, true)?);
        let upper = directed_multiply(first.0, first.1, false)?
            .max(directed_multiply(second.0, second.1, false)?)
            .max(directed_multiply(third.0, third.1, false)?)
            .max(directed_multiply(fourth.0, fourth.1, false)?);
        let same_sign =
            (self.lower >= 0.0 && other.lower >= 0.0) || (self.upper <= 0.0 && other.upper <= 0.0);
        let opposite_sign =
            (self.lower >= 0.0 && other.upper <= 0.0) || (self.upper <= 0.0 && other.lower >= 0.0);
        Some(Self {
            lower: if same_sign {
                lower.max(0.0)
            } else if self.contains(0.0) || other.contains(0.0) {
                lower.min(0.0)
            } else {
                lower
            },
            upper: if opposite_sign {
                upper.min(0.0)
            } else if self.contains(0.0) || other.contains(0.0) {
                upper.max(0.0)
            } else {
                upper
            },
        })
    }

    pub(super) fn divide(self, other: Self) -> Option<Self> {
        if other.contains(0.0) {
            return None;
        }
        if self.is_zero() {
            return Some(Self::point(0.0));
        }
        let quotients = [
            (self.lower, other.lower),
            (self.lower, other.upper),
            (self.upper, other.lower),
            (self.upper, other.upper),
        ];
        let [first, second, third, fourth] = quotients;
        let lower = directed_divide(first.0, first.1, true)?
            .min(directed_divide(second.0, second.1, true)?)
            .min(directed_divide(third.0, third.1, true)?)
            .min(directed_divide(fourth.0, fourth.1, true)?);
        let upper = directed_divide(first.0, first.1, false)?
            .max(directed_divide(second.0, second.1, false)?)
            .max(directed_divide(third.0, third.1, false)?)
            .max(directed_divide(fourth.0, fourth.1, false)?);
        let positive =
            (self.lower >= 0.0 && other.lower > 0.0) || (self.upper <= 0.0 && other.upper < 0.0);
        let negative =
            (self.lower >= 0.0 && other.upper < 0.0) || (self.upper <= 0.0 && other.lower > 0.0);
        Some(Self {
            lower: if positive { lower.max(0.0) } else { lower },
            upper: if negative { upper.min(0.0) } else { upper },
        })
    }

    pub(super) fn square_root(self) -> Option<Self> {
        if self.lower < 0.0 {
            return None;
        }
        Some(Self {
            lower: directed_sqrt(self.lower, true)?.max(0.0),
            upper: directed_sqrt(self.upper, false)?,
        })
    }

    fn negate(self) -> Self {
        Self {
            lower: -self.upper,
            upper: -self.lower,
        }
    }

    const fn is_zero(self) -> bool {
        self.lower == 0.0 && self.upper == 0.0
    }
}

fn directed_add(first: f64, second: f64, lower: bool) -> Option<f64> {
    let rounded = first + second;
    if !rounded.is_finite() {
        return None;
    }
    if first == -second {
        return Some(0.0);
    }
    let error = two_sum_error(first, second, rounded);
    if !error.is_finite() {
        return None;
    }
    let underflow_may_hide_error = rounded.abs() < f64::MIN_POSITIVE && error == 0.0;
    let bound = if underflow_may_hide_error && lower {
        rounded.next_down()
    } else if underflow_may_hide_error {
        rounded.next_up()
    } else if lower && error < 0.0 {
        rounded.next_down()
    } else if !lower && error > 0.0 {
        rounded.next_up()
    } else {
        rounded
    };
    bound.is_finite().then_some(bound)
}

fn two_sum_error(first: f64, second: f64, sum: f64) -> f64 {
    let second_virtual = sum - first;
    let first_virtual = sum - second_virtual;
    let second_roundoff = second - second_virtual;
    let first_roundoff = first - first_virtual;
    first_roundoff + second_roundoff
}

fn directed_multiply(first: f64, second: f64, lower: bool) -> Option<f64> {
    let rounded = first * second;
    if !rounded.is_finite() {
        return None;
    }
    if first == 0.0 || second == 0.0 {
        return Some(rounded);
    }
    let error = first.mul_add(second, -rounded);
    if !error.is_finite() {
        return None;
    }
    let underflow_may_hide_error = residual_may_underflow(error, rounded);
    let bound = if underflow_may_hide_error {
        if lower {
            rounded.next_down()
        } else {
            rounded.next_up()
        }
    } else if lower && error < 0.0 {
        rounded.next_down()
    } else if !lower && error > 0.0 {
        rounded.next_up()
    } else {
        rounded
    };
    bound.is_finite().then_some(bound)
}

fn directed_divide(numerator: f64, denominator: f64, lower: bool) -> Option<f64> {
    let rounded = numerator / denominator;
    if !rounded.is_finite() {
        return None;
    }
    if numerator == 0.0 {
        return Some(rounded);
    }
    let error = (-rounded).mul_add(denominator, numerator);
    if !error.is_finite() {
        return None;
    }
    let underflow_may_hide_error = residual_may_underflow(error, numerator);
    if error == 0.0 && !underflow_may_hide_error {
        return Some(rounded);
    }
    let quotient_error_positive = error.is_sign_positive() == denominator.is_sign_positive();
    let bound = if underflow_may_hide_error {
        if lower {
            rounded.next_down()
        } else {
            rounded.next_up()
        }
    } else if lower && !quotient_error_positive {
        rounded.next_down()
    } else if !lower && quotient_error_positive {
        rounded.next_up()
    } else {
        rounded
    };
    bound.is_finite().then_some(bound)
}

fn directed_sqrt(value: f64, lower: bool) -> Option<f64> {
    if value < 0.0 {
        return None;
    }
    if value == 0.0 {
        return Some(0.0);
    }
    let rounded = value.sqrt();
    if !rounded.is_finite() {
        return None;
    }
    let error = rounded.mul_add(rounded, -value);
    if !error.is_finite() {
        return None;
    }
    let underflow_may_hide_error = residual_may_underflow(error, value);
    let bound = if underflow_may_hide_error {
        if lower {
            rounded.next_down()
        } else {
            rounded.next_up()
        }
    } else if lower && error > 0.0 {
        rounded.next_down()
    } else if !lower && error < 0.0 {
        rounded.next_up()
    } else {
        rounded
    };
    bound.is_finite().then_some(bound)
}

fn residual_may_underflow(error: f64, scale: f64) -> bool {
    error == 0.0 && scale.abs() < RESIDUAL_UNDERFLOW_LIMIT
}

#[cfg(test)]
mod tests {
    use super::Interval;

    #[test]
    fn exact_zero_coefficients_do_not_spread_large_normal_coordinates() {
        let normal_coordinate = Interval::point(1.0e13);
        let horizontal_projection = Interval::point(0.0)
            .multiply(normal_coordinate)
            .expect("zero coefficient has a finite product");
        assert_eq!(horizontal_projection, Interval::point(0.0));
    }

    #[test]
    fn outward_operations_enclose_inexact_values_and_preserve_exact_axes() {
        let exact = Interval::point(1.0)
            .divide(Interval::point(1.0))
            .expect("unit division is finite");
        assert_eq!(exact, Interval::point(1.0));

        let rounded = Interval::point(1.0)
            .divide(Interval::point(10.0))
            .expect("finite quotient has an enclosure");
        assert_eq!(rounded.lower(), 0.1_f64.next_down());
        assert_eq!(rounded.upper(), 0.1_f64);
    }

    #[test]
    fn underflowing_squares_remain_nonnegative() {
        let smallest = f64::from_bits(1);
        let square = Interval::point(smallest)
            .multiply(Interval::point(smallest))
            .expect("finite subnormal product has an enclosure");
        assert!(square.lower() >= 0.0);
        assert!(square.upper() >= square.lower());
        let root = square
            .square_root()
            .expect("nonnegative subnormal enclosure has a root");
        assert!(root.contains(smallest));
    }

    #[test]
    fn exact_small_quotients_remain_enclosed() {
        let smallest = f64::from_bits(1);
        let ratio = Interval::point(smallest)
            .divide(Interval::point(smallest))
            .expect("finite subnormal quotient has an enclosure");
        assert!(ratio.contains(1.0));
    }
}
