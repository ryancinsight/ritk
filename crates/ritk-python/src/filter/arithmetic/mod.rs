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

macro_rules! binary_pyfn {
    ($name:ident, $filter:ident, $itk:literal, $doc:literal) => {
        #[doc = $doc]
        #[doc = ""]
        #[doc = concat!("ITK Parity: ", $itk)]
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &PyImage, b: &PyImage) -> RitkResult<PyImage> {
            let a_arc = std::sync::Arc::clone(&a.inner);
            let b_arc = std::sync::Arc::clone(&b.inner);
            py.allow_threads(|| {
                $filter::new()
                    .apply(a_arc.as_ref(), b_arc.as_ref())
                    .map_err(|e| RitkPyError::runtime(e.to_string()))
            })
            .map(into_py_image)
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
            let a_arc = std::sync::Arc::clone(&a.inner);
            let b_arc = std::sync::Arc::clone(&b.inner);
            let c_arc = std::sync::Arc::clone(&c.inner);
            py.allow_threads(|| {
                $filter::new()
                    .apply(a_arc.as_ref(), b_arc.as_ref(), c_arc.as_ref())
                    .map_err(|e| RitkPyError::runtime(e.to_string()))
            })
            .map(into_py_image)
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
