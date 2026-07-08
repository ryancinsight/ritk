macro_rules! unary_math_pyfn {
    ($name:ident, $filter:ident, $itk:literal, $doc:literal) => {
        #[doc = $doc]
        #[doc = ""]
        #[doc = concat!("ITK Parity: ", $itk)]
        #[pyfunction]
        pub fn $name(py: Python<'_>, image: &PyImage) -> PyImage {
            let arc = crate::image::py_image_to_burn(image);
            let out = py.allow_threads(|| $filter::new().apply(&arc));
            crate::image::burn_into_py_image(out)
        }
    };
}

macro_rules! binary_pyfn {
    ($name:ident, $filter:ident, $itk:literal, $doc:literal) => {
        #[doc = $doc]
        #[doc = ""]
        #[doc = concat!("ITK Parity: ", $itk)]
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &PyImage, b: &PyImage) -> RitkResult<PyImage> {
            let a_arc = crate::image::py_image_to_burn(a);
            let b_arc = crate::image::py_image_to_burn(b);
            py.allow_threads(|| {
                $filter::new()
                    .apply(&a_arc, &b_arc)
                    .map_err(|e| RitkPyError::runtime(e.to_string()))
            })
            .map(crate::image::burn_into_py_image)
        }
    };
}

macro_rules! ternary_pyfn {
    ($name:ident, $filter:ident, $itk:literal, $doc:literal) => {
        #[doc = $doc]
        #[doc = ""]
        #[doc = concat!("ITK Parity: ", $itk)]
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &PyImage, b: &PyImage, c: &PyImage) -> RitkResult<PyImage> {
            let a_arc = crate::image::py_image_to_burn(a);
            let b_arc = crate::image::py_image_to_burn(b);
            let c_arc = crate::image::py_image_to_burn(c);
            py.allow_threads(|| {
                $filter::new()
                    .apply(&a_arc, &b_arc, &c_arc)
                    .map_err(|e| RitkPyError::runtime(e.to_string()))
            })
            .map(crate::image::burn_into_py_image)
        }
    };
}

mod binary;
mod mask;
mod ternary;
mod unary;

pub use binary::*;
pub use mask::*;
pub use ternary::*;
pub use unary::*;
