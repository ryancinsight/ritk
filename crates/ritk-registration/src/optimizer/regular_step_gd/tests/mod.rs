//! RSGD test suite root: shared test fixtures and submodule declarations.

pub(super) mod config;
pub(super) mod functional;
pub(super) mod invariants;

use burn_ndarray::NdArray;
use ritk_image::burn::backend::Autodiff;
use ritk_image::burn::module::{Module, Param};
use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;
use ritk_image::tensor::;

pub(super) type TestBackend = Autodiff<NdArray<f32>>;

// â”€â”€ Shared test module: f(Î¸) = Î£áµ¢ Î¸áµ¢Â² â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Minimal 1-D parameter module with analytical gradient âˆ‡f = 2Î¸.
// Used across all RSGD step-mechanic tests.
#[derive(Module, Debug)]
pub(super) struct Quadratic<B: Backend> {
    pub(super) x: Param<Tensor<f32, B>> }

impl<B: Backend> Quadratic<B> {
    pub(super) fn new(x0: &[f32], device: &B::Device) -> Self {
        let x = Tensor::<f32, B>::from_data(::from(x0), device);
        Self {
            x: Param::from_tensor(x) }
    }

    /// f(Î¸) = Î£áµ¢ Î¸áµ¢Â² (autodiff-tracked)
    pub(super) fn forward(&self) -> Tensor<f32, B> {
        let x = self.x.val();
        x.clone() * x
    }

    /// L = Î£áµ¢ Î¸áµ¢Â² (scalar, no autodiff)
    pub(super) fn loss_value(&self) -> f64 {
        let data = self.x.val().to_data();
        let slice = data.as_slice::<f32>().unwrap();
        slice.iter().map(|&v| (v as f64) * (v as f64)).sum()
    }

    /// First element of the parameter vector.
    pub(super) fn param_value(&self) -> f64 {
        let data = self.x.val().to_data();
        let slice = data.as_slice::<f32>().unwrap();
        slice[0] as f64
    }
}
