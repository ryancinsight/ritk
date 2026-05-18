//! RSGD test suite root: shared test fixtures and submodule declarations.

pub(super) mod config;
pub(super) mod functional;
pub(super) mod invariants;

use burn::backend::Autodiff;
use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_ndarray::NdArray;

pub(super) type TestBackend = Autodiff<NdArray<f32>>;

// ── Shared test module: f(θ) = Σᵢ θᵢ² ─────────────────────────────────────
//
// Minimal 1-D parameter module with analytical gradient ∇f = 2θ.
// Used across all RSGD step-mechanic tests.
#[derive(Module, Debug)]
pub(super) struct Quadratic<B: Backend> {
    pub(super) x: Param<Tensor<B, 1>>,
}

impl<B: Backend> Quadratic<B> {
    pub(super) fn new(x0: &[f32], device: &B::Device) -> Self {
        let x = Tensor::<B, 1>::from_data(burn::tensor::TensorData::from(x0), device);
        Self {
            x: Param::from_tensor(x),
        }
    }

    /// f(θ) = Σᵢ θᵢ² (autodiff-tracked)
    pub(super) fn forward(&self) -> Tensor<B, 1> {
        let x = self.x.val();
        x.clone() * x
    }

    /// L = Σᵢ θᵢ² (scalar, no autodiff)
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
