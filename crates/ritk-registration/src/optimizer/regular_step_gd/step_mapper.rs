//! RSGD step mapper: applies `θ_new = θ − (Δ / ‖g‖) · g`.

use burn::module::Param;
use burn::optim::GradientsParams;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use std::marker::PhantomData;

/// `ModuleMapper` that applies the RSGD update rule on an AutodiffModule:
///
/// ```text
/// θ_new = θ_old − (step_length / grad_norm) · g
/// ```
///
/// This is mathematically equivalent to stepping by `step_length` in the
/// unit negative-gradient direction ĝ = g / ‖g‖₂.
///
/// The mapper reads gradients from `GradientsParams` (consuming them via
/// `remove`), scales by `effective_lr = step_length / grad_norm`, and
/// subtracts from the parameter tensor. Operates on the inner (non-autodiff)
/// backend tensors, then wraps the result back into the autodiff tensor.
pub(super) struct RsgdStepMapper<'a, B: AutodiffBackend> {
    pub(super) grads: &'a mut GradientsParams,
    pub(super) effective_lr: f64,
    pub(super) _phantom: PhantomData<fn() -> B>,
}

impl<'a, B: AutodiffBackend> RsgdStepMapper<'a, B> {
    pub(super) fn new(grads: &'a mut GradientsParams, effective_lr: f64) -> Self {
        Self {
            grads,
            effective_lr,
            _phantom: PhantomData,
        }
    }
}

impl<B: AutodiffBackend> burn::module::ModuleMapper<B> for RsgdStepMapper<'_, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let is_require_grad = tensor.is_require_grad();

        let tensor = if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(id) {
            let inner_tensor = tensor.inner();
            let delta = grad.mul_scalar(self.effective_lr);
            let updated = inner_tensor.sub(delta);
            let mut updated = Tensor::<B, D>::from_inner(updated);
            if is_require_grad {
                updated = updated.require_grad();
            }
            updated
        } else {
            tensor
        };

        Param::from_mapped_value(id, tensor, mapper)
    }
}
