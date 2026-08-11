macro_rules! unary_math_pyfn {
    ($name:ident, $filter:ident, $itk:literal, $doc:literal) => {
        #[doc = $doc]
        #[doc = ""]
        #[doc = concat!("ITK Parity: ", $itk)]
        #[pyfunction]
        pub fn $name(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
            let native = std::sync::Arc::clone(&image.inner);
            let backend = coeus_core::MoiraiBackend;
            py.allow_threads(|| {
                $filter::new()
                    .apply_native(native.as_ref(), &backend)
                    .map_err(|e| RitkPyError::runtime(e.to_string()))
            })
            .map(crate::image::into_py_image)
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
            let a_native = std::sync::Arc::clone(&a.inner);
            let b_native = std::sync::Arc::clone(&b.inner);
            let backend = coeus_core::MoiraiBackend;
            py.allow_threads(|| {
                $filter::new()
                    .apply_native(a_native.as_ref(), b_native.as_ref(), &backend)
                    .map_err(|e| RitkPyError::runtime(e.to_string()))
            })
            .map(crate::image::into_py_image)
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
            let a_native = std::sync::Arc::clone(&a.inner);
            let b_native = std::sync::Arc::clone(&b.inner);
            let c_native = std::sync::Arc::clone(&c.inner);
            let backend = coeus_core::MoiraiBackend;
            py.allow_threads(|| {
                $filter::new()
                    .apply_native(
                        a_native.as_ref(),
                        b_native.as_ref(),
                        c_native.as_ref(),
                        &backend,
                    )
                    .map_err(|e| RitkPyError::runtime(e.to_string()))
            })
            .map(crate::image::into_py_image)
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
