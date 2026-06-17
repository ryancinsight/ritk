//! Pixelwise binary image arithmetic filters.
//!
//! Each function combines two co-registered images of identical shape via a
//! pointwise binary operation applied independently to every voxel.
//! Spatial metadata (origin, spacing, direction) is taken from the first input image.
//!
//! ITK Parity: AddImageFilter, SubtractImageFilter, MultiplyImageFilter,
//!             DivideImageFilter, MinimumImageFilter, MaximumImageFilter

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::{
    AbsImageFilter, AbsoluteValueDifferenceImageFilter, AcosImageFilter, AddImageFilter,
    AsinImageFilter, Atan2ImageFilter, AtanImageFilter, BoundedReciprocalImageFilter,
    ClampImageFilter, CosImageFilter, DivideImageFilter, ExpImageFilter, ExpNegativeImageFilter,
    ImageMaxFilter, ImageMinFilter, InvertIntensityFilter, Log10ImageFilter, LogImageFilter,
    MaskImageFilter, MaskNegatedImageFilter, MultiplyImageFilter, PowImageFilter, SinImageFilter,
    SqrtImageFilter, SquareImageFilter, SquaredDifferenceImageFilter, SubtractImageFilter,
    TanImageFilter,
};

/// Pixelwise clamp to `[lower, upper]`. ITK Parity: ClampImageFilter.
#[pyfunction]
#[pyo3(signature = (image, lower, upper))]
pub fn clamp_image(py: Python<'_>, image: &PyImage, lower: f32, upper: f32) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        ClampImageFilter::new(lower, upper)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Invert intensities about `maximum`: out(x) = maximum − in(x).
/// ITK Parity: InvertIntensityImageFilter.
#[pyfunction]
#[pyo3(signature = (image, maximum = 255.0))]
pub fn invert_intensity(py: Python<'_>, image: &PyImage, maximum: f32) -> PyImage {
    let arc = std::sync::Arc::clone(&image.inner);
    let out = py.allow_threads(|| InvertIntensityFilter::with_maximum(maximum).apply(arc.as_ref()));
    into_py_image(out)
}

/// Mask `image` by `mask`: keep where mask > 0, else `outside_value`.
/// ITK Parity: MaskImageFilter.
#[pyfunction]
#[pyo3(signature = (image, mask, outside_value = 0.0))]
pub fn mask_image(
    py: Python<'_>,
    image: &PyImage,
    mask: &PyImage,
    outside_value: f32,
) -> RitkResult<PyImage> {
    let img = std::sync::Arc::clone(&image.inner);
    let msk = std::sync::Arc::clone(&mask.inner);
    py.allow_threads(|| {
        MaskImageFilter::new()
            .with_outside_value(outside_value)
            .apply(img.as_ref(), msk.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Mask `image` by the negation of `mask`: keep where mask ≤ 0, else
/// `outside_value`. ITK Parity: MaskNegatedImageFilter.
#[pyfunction]
#[pyo3(signature = (image, mask, outside_value = 0.0))]
pub fn mask_negated_image(
    py: Python<'_>,
    image: &PyImage,
    mask: &PyImage,
    outside_value: f32,
) -> RitkResult<PyImage> {
    let img = std::sync::Arc::clone(&image.inner);
    let msk = std::sync::Arc::clone(&mask.inner);
    py.allow_threads(|| {
        MaskNegatedImageFilter::new()
            .with_outside_value(outside_value)
            .apply(img.as_ref(), msk.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Generate a pixelwise unary-math `#[pyfunction]` that mirrors an ITK unary
/// math image filter. Each applies `Filter::new().apply` to the input under
/// `allow_threads`; the operation is infallible (pure elementwise map).
macro_rules! unary_math_pyfn {
    ($name:ident, $filter:ident, $itk:literal, $doc:literal) => {
        #[doc = $doc]
        #[doc = ""]
        #[doc = concat!("ITK Parity: ", $itk)]
        #[pyfunction]
        pub fn $name(py: Python<'_>, image: &PyImage) -> PyImage {
            let arc = std::sync::Arc::clone(&image.inner);
            let out = py.allow_threads(|| $filter::new().apply(arc.as_ref()));
            into_py_image(out)
        }
    };
}

unary_math_pyfn!(abs_image, AbsImageFilter, "AbsImageFilter", "Pixelwise absolute value: out(x) = |in(x)|.");
unary_math_pyfn!(sqrt_image, SqrtImageFilter, "SqrtImageFilter", "Pixelwise square root: out(x) = sqrt(in(x)).");
unary_math_pyfn!(square_image, SquareImageFilter, "SquareImageFilter", "Pixelwise square: out(x) = in(x)^2.");
unary_math_pyfn!(exp_image, ExpImageFilter, "ExpImageFilter", "Pixelwise exponential: out(x) = exp(in(x)).");
unary_math_pyfn!(log_image, LogImageFilter, "LogImageFilter", "Pixelwise natural log: out(x) = ln(in(x)).");
unary_math_pyfn!(log10_image, Log10ImageFilter, "Log10ImageFilter", "Pixelwise base-10 log: out(x) = log10(in(x)).");
unary_math_pyfn!(exp_negative_image, ExpNegativeImageFilter, "ExpNegativeImageFilter", "Pixelwise negative exponential: out(x) = exp(-in(x)).");
unary_math_pyfn!(sin_image, SinImageFilter, "SinImageFilter", "Pixelwise sine: out(x) = sin(in(x)).");
unary_math_pyfn!(cos_image, CosImageFilter, "CosImageFilter", "Pixelwise cosine: out(x) = cos(in(x)).");
unary_math_pyfn!(tan_image, TanImageFilter, "TanImageFilter", "Pixelwise tangent: out(x) = tan(in(x)).");
unary_math_pyfn!(asin_image, AsinImageFilter, "AsinImageFilter", "Pixelwise arcsine: out(x) = asin(in(x)).");
unary_math_pyfn!(acos_image, AcosImageFilter, "AcosImageFilter", "Pixelwise arccosine: out(x) = acos(in(x)).");
unary_math_pyfn!(atan_image, AtanImageFilter, "AtanImageFilter", "Pixelwise arctangent: out(x) = atan(in(x)).");
unary_math_pyfn!(bounded_reciprocal_image, BoundedReciprocalImageFilter, "BoundedReciprocalImageFilter", "Pixelwise bounded reciprocal: out(x) = 1 / (1 + in(x)).");

/// Pixelwise addition: out(x) = a(x) + b(x).
///
/// ITK Parity: AddImageFilter
#[pyfunction]
pub fn add_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> RitkResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    py.allow_threads(|| {
        AddImageFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Pixelwise subtraction: out(x) = a(x) - b(x).
///
/// ITK Parity: SubtractImageFilter
#[pyfunction]
pub fn subtract_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> RitkResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    py.allow_threads(|| {
        SubtractImageFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Pixelwise multiplication: out(x) = a(x) * b(x).
///
/// ITK Parity: MultiplyImageFilter
#[pyfunction]
pub fn multiply_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> RitkResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    py.allow_threads(|| {
        MultiplyImageFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Pixelwise division: out(x) = a(x) / b(x).
///
/// Matches ITK/SimpleITK `DivideImageFilter` for all non-zero denominators.
/// Division-by-zero convention differs: ritk yields 0 (a safer default for
/// downstream computation), whereas ITK yields the maximum float value
/// (`NumericTraits::max()`, ≈ 3.4e38). Pre-mask zeros if exact ITK behaviour is
/// required.
#[pyfunction]
pub fn divide_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> RitkResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    py.allow_threads(|| {
        DivideImageFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Pixelwise squared difference: out(x) = (a(x) - b(x))^2.
///
/// ITK Parity: SquaredDifferenceImageFilter
#[pyfunction]
pub fn squared_difference_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> RitkResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    py.allow_threads(|| {
        SquaredDifferenceImageFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Pixelwise absolute difference: out(x) = |a(x) - b(x)|.
///
/// ITK Parity: AbsoluteValueDifferenceImageFilter
#[pyfunction]
pub fn absolute_value_difference_images(
    py: Python<'_>,
    a: &PyImage,
    b: &PyImage,
) -> RitkResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    py.allow_threads(|| {
        AbsoluteValueDifferenceImageFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Pixelwise four-quadrant arctangent: out(x) = atan2(a(x), b(x)).
///
/// ITK Parity: Atan2ImageFilter
#[pyfunction]
pub fn atan2_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> RitkResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    py.allow_threads(|| {
        Atan2ImageFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Pixelwise power: out(x) = a(x) ^ b(x).
///
/// ITK Parity: PowImageFilter
#[pyfunction]
pub fn pow_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> RitkResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    py.allow_threads(|| {
        PowImageFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Pixelwise minimum: out(x) = min(a(x), b(x)).
///
/// ITK Parity: MinimumImageFilter
#[pyfunction]
pub fn minimum_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> RitkResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    py.allow_threads(|| {
        ImageMinFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Pixelwise maximum: out(x) = max(a(x), b(x)).
///
/// ITK Parity: MaximumImageFilter
#[pyfunction]
pub fn maximum_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> RitkResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    py.allow_threads(|| {
        ImageMaxFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
