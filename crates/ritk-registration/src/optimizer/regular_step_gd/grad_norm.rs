//! Gradient norm visitor for computing ‖∇L‖₂ across module parameters.

use ritk_image::burn::module::Param;
use ritk_image::burn::optim::GradientsParams;
use ritk_image::tensor::AutodiffBackend;
use ritk_image::tensor::{ElementConversion, Tensor};
use std::marker::PhantomData;

/// `ModuleVisitor` that accumulates the squared-L2-norm of all gradient tensors
/// stored in a [`GradientsParams`], producing the global gradient norm
/// ‖∇L‖₂ = √(Σᵢ ‖gᵢ‖₂²) across all parameter tensors.
///
/// Type parameter `B` is `AutodiffBackend`. Gradients are retrieved on
/// `B::InnerBackend` (the non-autodiff backend) since `GradientsParams`
/// stores inner-backend tensors.
pub(super) struct GradientNormVisitor<'a, B: AutodiffBackend> {
    pub(super) grads: &'a GradientsParams,
    pub(super) norm_sq: f64,
    pub(super) _phantom: PhantomData<fn() -> B>,
}

impl<'a, B: AutodiffBackend> GradientNormVisitor<'a, B> {
    pub(super) fn new(grads: &'a GradientsParams) -> Self {
        Self {
            grads,
            norm_sq: 0.0,
            _phantom: PhantomData,
        }
    }

    pub(super) fn into_norm(self) -> f64 {
        self.norm_sq.sqrt()
    }
}

impl<B: AutodiffBackend> ritk_image::burn::module::ModuleVisitor<B> for GradientNormVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(grad) = self.grads.get::<B::InnerBackend, D>(param.id) {
            let data = grad.to_data();
            if let Ok(slice) = data.as_slice::<f32>() {
                for &v in slice {
                    let vf: f64 = v.elem();
                    self.norm_sq += vf * vf;
                }
            } else if let Ok(slice) = data.as_slice::<f64>() {
                for &v in slice {
                    self.norm_sq += v * v;
                }
            }
        }
    }
}
