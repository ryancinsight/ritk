use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::{
    AbsoluteValueDifferenceImageFilter, AddImageFilter, AndImageFilter, Atan2ImageFilter,
    BinaryMagnitudeImageFilter, DivideFloorImageFilter, DivideImageFilter, DivideRealImageFilter,
    EqualImageFilter, GreaterEqualImageFilter, GreaterImageFilter, ImageMaxFilter, ImageMinFilter,
    LessEqualImageFilter, LessImageFilter, MultiplyImageFilter, NotEqualImageFilter,
    OrImageFilter, PowImageFilter, SquaredDifferenceImageFilter, SubtractImageFilter, XorImageFilter,
};

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

/// Pixelwise magnitude: out(x) = sqrt(a(x)^2 + b(x)^2).
///
/// ITK Parity: BinaryMagnitudeImageFilter
#[pyfunction]
pub fn binary_magnitude_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> RitkResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    py.allow_threads(|| {
        BinaryMagnitudeImageFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Fold a binary image operation over a non-empty list of co-shaped images
/// (`acc = op(op(op(i0, i1), i2), …)`). Used by the N-ary `Add` / `Maximum`.
fn nary_fold<Op: ritk_filter::BinaryOp>(
    py: Python<'_>,
    images: Vec<Py<PyImage>>,
    what: &str,
) -> RitkResult<PyImage> {
    if images.is_empty() {
        return Err(RitkPyError::value(format!(
            "{what}: needs at least one image"
        )));
    }
    let arcs: Vec<_> = images
        .iter()
        .map(|p| std::sync::Arc::clone(&p.bind(py).borrow().inner))
        .collect();
    py.allow_threads(|| {
        let mut acc = (*arcs[0]).clone();
        for img in &arcs[1..] {
            acc = ritk_filter::BinaryOpFilter::<Op>::new()
                .apply(&acc, img.as_ref())
                .map_err(|e| RitkPyError::runtime(e.to_string()))?;
        }
        Ok(acc)
    })
    .map(into_py_image)
}

/// Pixelwise sum of any number of images: `out(x) = Σ_i img_i(x)`.
///
/// ITK Parity: NaryAddImageFilter (`sitk.NaryAdd`)
#[pyfunction]
pub fn nary_add(py: Python<'_>, images: Vec<Py<PyImage>>) -> RitkResult<PyImage> {
    nary_fold::<ritk_filter::AddOp>(py, images, "nary_add")
}

/// Pixelwise maximum of any number of images: `out(x) = max_i img_i(x)`.
///
/// ITK Parity: NaryMaximumImageFilter (`sitk.NaryMaximum`)
#[pyfunction]
pub fn nary_maximum(py: Python<'_>, images: Vec<Py<PyImage>>) -> RitkResult<PyImage> {
    nary_fold::<ritk_filter::MaxOp>(py, images, "nary_maximum")
}

binary_pyfn!(
    divide_real_images,
    DivideRealImageFilter,
    "DivideRealImageFilter",
    "Pixelwise real division: out(x) = a/b (FLT_MAX where b==0)."
);
binary_pyfn!(
    divide_floor_images,
    DivideFloorImageFilter,
    "DivideFloorImageFilter",
    "Pixelwise floored division: out(x) = floor(a/b) (FLT_MAX where b==0)."
);

binary_pyfn!(
    equal_images,
    EqualImageFilter,
    "EqualImageFilter",
    "Pixelwise equality mask: out(x) = 1 where a(x) == b(x), else 0."
);
binary_pyfn!(
    not_equal_images,
    NotEqualImageFilter,
    "NotEqualImageFilter",
    "Pixelwise inequality mask: out(x) = 1 where a(x) != b(x), else 0."
);
binary_pyfn!(
    greater_images,
    GreaterImageFilter,
    "GreaterImageFilter",
    "Pixelwise greater-than mask: out(x) = 1 where a(x) > b(x), else 0."
);
binary_pyfn!(
    greater_equal_images,
    GreaterEqualImageFilter,
    "GreaterEqualImageFilter",
    "Pixelwise greater-or-equal mask: out(x) = 1 where a(x) >= b(x), else 0."
);
binary_pyfn!(
    less_images,
    LessImageFilter,
    "LessImageFilter",
    "Pixelwise less-than mask: out(x) = 1 where a(x) < b(x), else 0."
);
binary_pyfn!(
    less_equal_images,
    LessEqualImageFilter,
    "LessEqualImageFilter",
    "Pixelwise less-or-equal mask: out(x) = 1 where a(x) <= b(x), else 0."
);
binary_pyfn!(
    and_images,
    AndImageFilter,
    "AndImageFilter",
    "Pixelwise logical AND of binary masks: out(x) = 1 where a>0 and b>0, else 0."
);
binary_pyfn!(
    or_images,
    OrImageFilter,
    "OrImageFilter",
    "Pixelwise logical OR of binary masks: out(x) = 1 where a>0 or b>0, else 0."
);
binary_pyfn!(
    xor_images,
    XorImageFilter,
    "XorImageFilter",
    "Pixelwise logical XOR of binary masks: out(x) = 1 where exactly one of a,b > 0, else 0."
);

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
