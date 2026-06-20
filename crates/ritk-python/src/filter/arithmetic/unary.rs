use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::{
    AbsImageFilter, AcosImageFilter, AsinImageFilter, AtanImageFilter, BinaryNotImageFilter,
    ClampImageFilter, CosImageFilter, ExpImageFilter, ExpNegativeImageFilter, InvertIntensityFilter,
    Log10ImageFilter, LogImageFilter, ModulusImageFilter, NotImageFilter, RoundImageFilter,
    SinImageFilter, SqrtImageFilter, SquareImageFilter, TanImageFilter, UnaryMinusImageFilter,
    BoundedReciprocalImageFilter,
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

/// Pixelwise modulo: out(x) = in(x) % dividend (integer images; C truncated
/// remainder). ITK Parity: ModulusImageFilter (`sitk.Modulus`).
#[pyfunction]
#[pyo3(signature = (image, dividend))]
pub fn modulus(py: Python<'_>, image: &PyImage, dividend: i64) -> RitkResult<PyImage> {
    if dividend == 0 {
        return Err(RitkPyError::value("modulus: dividend must be non-zero"));
    }
    let arc = std::sync::Arc::clone(&image.inner);
    let out = py.allow_threads(|| ModulusImageFilter::new(dividend).apply(arc.as_ref()));
    Ok(into_py_image(out))
}

/// Binary logical NOT: out(x) = background where in(x) == foreground, else
/// foreground. ITK Parity: BinaryNotImageFilter.
#[pyfunction]
#[pyo3(signature = (image, foreground=1.0, background=0.0))]
pub fn binary_not(
    py: Python<'_>,
    image: &PyImage,
    foreground: f32,
    background: f32,
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let out = py.allow_threads(|| {
        BinaryNotImageFilter::with_labels(foreground, background).apply(arc.as_ref())
    });
    Ok(into_py_image(out))
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

unary_math_pyfn!(
    abs_image,
    AbsImageFilter,
    "AbsImageFilter",
    "Pixelwise absolute value: out(x) = |in(x)|."
);
unary_math_pyfn!(
    sqrt_image,
    SqrtImageFilter,
    "SqrtImageFilter",
    "Pixelwise square root: out(x) = sqrt(in(x))."
);
unary_math_pyfn!(
    square_image,
    SquareImageFilter,
    "SquareImageFilter",
    "Pixelwise square: out(x) = in(x)^2."
);
unary_math_pyfn!(
    exp_image,
    ExpImageFilter,
    "ExpImageFilter",
    "Pixelwise exponential: out(x) = exp(in(x))."
);
unary_math_pyfn!(
    log_image,
    LogImageFilter,
    "LogImageFilter",
    "Pixelwise natural log: out(x) = ln(in(x))."
);
unary_math_pyfn!(
    log10_image,
    Log10ImageFilter,
    "Log10ImageFilter",
    "Pixelwise base-10 log: out(x) = log10(in(x))."
);
unary_math_pyfn!(
    exp_negative_image,
    ExpNegativeImageFilter,
    "ExpNegativeImageFilter",
    "Pixelwise negative exponential: out(x) = exp(-in(x))."
);
unary_math_pyfn!(
    sin_image,
    SinImageFilter,
    "SinImageFilter",
    "Pixelwise sine: out(x) = sin(in(x))."
);
unary_math_pyfn!(
    cos_image,
    CosImageFilter,
    "CosImageFilter",
    "Pixelwise cosine: out(x) = cos(in(x))."
);
unary_math_pyfn!(
    tan_image,
    TanImageFilter,
    "TanImageFilter",
    "Pixelwise tangent: out(x) = tan(in(x))."
);
unary_math_pyfn!(
    asin_image,
    AsinImageFilter,
    "AsinImageFilter",
    "Pixelwise arcsine: out(x) = asin(in(x))."
);
unary_math_pyfn!(
    acos_image,
    AcosImageFilter,
    "AcosImageFilter",
    "Pixelwise arccosine: out(x) = acos(in(x))."
);
unary_math_pyfn!(
    atan_image,
    AtanImageFilter,
    "AtanImageFilter",
    "Pixelwise arctangent: out(x) = atan(in(x))."
);
unary_math_pyfn!(
    bounded_reciprocal_image,
    BoundedReciprocalImageFilter,
    "BoundedReciprocalImageFilter",
    "Pixelwise bounded reciprocal: out(x) = 1 / (1 + in(x))."
);
unary_math_pyfn!(
    unary_minus_image,
    UnaryMinusImageFilter,
    "UnaryMinusImageFilter",
    "Pixelwise negation: out(x) = -in(x)."
);
unary_math_pyfn!(
    round_image,
    RoundImageFilter,
    "RoundImageFilter",
    "Pixelwise round to nearest integer (half-up): out(x) = floor(in(x) + 0.5)."
);
unary_math_pyfn!(
    not_image,
    NotImageFilter,
    "NotImageFilter",
    "Pixelwise logical NOT of a mask: out(x) = 1 where in(x) == 0, else 0."
);
