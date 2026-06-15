//! Global Normalized Cross Correlation (NCC) loss module.

use burn::{
    module::{Ignored, Module},
    tensor::{backend::Backend, Tensor},
};
use std::marker::PhantomData;

/// Global Normalized Cross Correlation (NCC) Loss.
#[derive(Module, Debug)]
pub struct GlobalNCCLoss<B: Backend> {
    pub(super) epsilon: Ignored<f32>,
    // BURN-derive: must stay `PhantomData<B>` (invariant). Burn 0.19.1 only
    // implements `Module<B> for PhantomData<B>` (see
    // burn-core/src/module/param/constant.rs:202), not for the covariant
    // `PhantomData<fn() -> B>` form. An earlier attempt to switch to the
    // covariant form failed to compile with 108 errors; see audit Sprint
    // 374-08 and the compile error against `burn-core-0.19.1`. The blocked
    // variance relaxation is tracked in `backlog.md` §Blocked/Deferred.
    pub(super) phantom: PhantomData<B>,
}

impl<B: Backend> Default for GlobalNCCLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> GlobalNCCLoss<B> {
    pub fn new() -> Self {
        Self {
            epsilon: Ignored(1e-5),
            phantom: PhantomData,
        }
    }

    pub fn forward(&self, y_true: Tensor<B, 5>, y_pred: Tensor<B, 5>) -> Tensor<B, 1> {
        let i_mean = y_true.clone().mean().reshape([1, 1, 1, 1, 1]);
        let j_mean = y_pred.clone().mean().reshape([1, 1, 1, 1, 1]);

        let i_hat = y_true.clone().sub(i_mean);
        let j_hat = y_pred.clone().sub(j_mean);

        let num = (i_hat.clone() * j_hat.clone()).mean();
        let den =
            (i_hat.powf_scalar(2.0).mean() * j_hat.powf_scalar(2.0).mean() + *self.epsilon).sqrt();

        num.div(den).neg().reshape([1])
    }
}
