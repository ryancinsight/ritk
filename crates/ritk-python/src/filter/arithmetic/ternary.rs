use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::{
    TernaryAddImageFilter, TernaryMagnitudeImageFilter, TernaryMagnitudeSquaredImageFilter,
};

ternary_pyfn!(
    ternary_add_images,
    TernaryAddImageFilter,
    "TernaryAddImageFilter",
    "Pixelwise sum of three images: out(x) = a + b + c."
);
ternary_pyfn!(
    ternary_magnitude_images,
    TernaryMagnitudeImageFilter,
    "TernaryMagnitudeImageFilter",
    "Pixelwise magnitude of three images: out(x) = sqrt(a^2 + b^2 + c^2)."
);
ternary_pyfn!(
    ternary_magnitude_squared_images,
    TernaryMagnitudeSquaredImageFilter,
    "TernaryMagnitudeSquaredImageFilter",
    "Pixelwise squared magnitude of three images: out(x) = a^2 + b^2 + c^2."
);
