//! First-order gradient regularization loss for deformation fields.

use burn::{
    module::{Ignored, Module},
    tensor::{backend::Backend, Tensor},
};
use std::marker::PhantomData;

/// Penalty type for gradient-based regularization.
#[derive(Debug, Clone, Copy, Default)]
pub enum GradientPenalty {
    #[default]
    L2,
    L1,
}

/// First-order gradient regularization (Sobolev smoothness penalty on deformation fields).
///
/// Enforces smoothness by penalizing high-frequency spatial gradients:
/// R(φ) = (1/|Ω|) ∫ ‖∇φ(x)‖ₚᵖ dx
#[derive(Module, Debug)]
pub struct GradLoss<B: Backend> {
    pub(super) penalty: Ignored<GradientPenalty>,
    // BURN-forced: Module derive requires invariant B for correct gradient tracking.
    pub(super) phantom: PhantomData<B>,
}

impl<B: Backend> GradLoss<B> {
    pub fn new(penalty: GradientPenalty) -> Self {
        Self {
            penalty: Ignored(penalty),
            phantom: PhantomData,
        }
    }

    // NOTE: 5 clones required by burn's slice(self, ...) ownership model.
    // Each clone feeds a separate finite-difference slice. Will be eliminable
    // when burn adds slice_ref(&self, ...).
    pub fn forward(&self, flow: Tensor<B, 5>) -> Tensor<B, 1> {
        let [b, c, d, h, w] = flow.dims();
        let dy = flow.clone().slice([0..b, 0..c, 1..d, 0..h, 0..w])
            - flow.clone().slice([0..b, 0..c, 0..d - 1, 0..h, 0..w]);
        let dx = flow.clone().slice([0..b, 0..c, 0..d, 1..h, 0..w])
            - flow.clone().slice([0..b, 0..c, 0..d, 0..h - 1, 0..w]);
        let dz = flow.clone().slice([0..b, 0..c, 0..d, 0..h, 1..w])
            - flow.clone().slice([0..b, 0..c, 0..d, 0..h, 0..w - 1]);

        match *self.penalty {
            GradientPenalty::L2 => {
                let loss = (dy.powf_scalar(2.0).mean()
                    + dx.powf_scalar(2.0).mean()
                    + dz.powf_scalar(2.0).mean())
                    / 3.0;
                loss.reshape([1])
            }
            GradientPenalty::L1 => {
                let loss = (dy.abs().mean() + dx.abs().mean() + dz.abs().mean()) / 3.0;
                loss.reshape([1])
            }
        }
    }
}
