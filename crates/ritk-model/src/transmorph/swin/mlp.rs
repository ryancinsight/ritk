//! Swin-block MLP, Coeus-native.
//!
//! A two-layer feed-forward network (`Linear → GELU → Linear`) applied along the
//! channel (last) axis of a `[B, D, H, W, C]` token volume. Built on
//! [`coeus_nn::Linear`] over [`coeus_autograd::Var`] with the exact GELU
//! activation; gradients flow to both linear layers through the autograd graph.

use crate::error::ModelError;
use coeus_autograd::{gelu, Parameter, Var};
use coeus_core::{Backend, CpuAddressableStorageMut};
use coeus_nn::module::Module;
use coeus_nn::Linear;
use coeus_ops::{BackendOps, CpuBackend};

/// Two-layer channel-wise MLP with a GELU nonlinearity.
#[derive(Clone)]
pub struct Mlp<B: Backend + BackendOps<f32> + Default> {
    fc1: Linear<f32, B>,
    fc2: Linear<f32, B>,
}

impl<B> Mlp<B>
where
    B: Backend + BackendOps<f32> + Default + CpuBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorageMut<f32>,
{
    /// Construct an MLP mapping `input_dim → hidden_dim → input_dim`.
    ///
    /// Weights are Kaiming-uniform-initialized (fan-in of each layer), biases
    /// zero.
    ///
    /// This used to build each layer and then re-initialize it, because
    /// `Linear::new` left every weight at 1.0 and the model needs the
    /// non-degenerate scheme. `Linear::with_seed` does it directly now
    /// (coeus ADR 0067).
    ///
    /// # Panics
    ///
    /// Panics when a layer dimension is zero, matching
    /// [`super::attention::WindowAttention::new`] -- both are constructed from
    /// a config this crate validates before it reaches them.
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let fc1 = Linear::with_seed(input_dim, hidden_dim, true, seed)
            .expect("invariant: MLP input fan is positive");
        let fc2 = Linear::with_seed(hidden_dim, input_dim, true, seed ^ 0x5DEE_CE66)
            .expect("invariant: MLP hidden fan is positive");
        Self { fc1, fc2 }
    }
}

impl<B> Mlp<B>
where
    B: Backend + BackendOps<f32> + Default,
{
    /// Forward pass over a `[B, D, H, W, C]` token volume.
    pub fn forward(&self, x: &Var<f32, B>) -> Result<Var<f32, B>, ModelError> {
        let x = self.fc1.forward(x)?;
        let x = gelu(&x);
        Ok(self.fc2.forward(&x)?)
    }

    /// Trainable parameters in forward order.
    pub fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }

    /// Trainable parameters with stable hierarchical names.
    pub fn named_parameters(&self) -> Vec<Parameter<f32, B>> {
        let mut named: Vec<Parameter<f32, B>> = self
            .fc1
            .named_parameters()
            .into_iter()
            .map(|p| p.with_prefix("fc1"))
            .collect();
        named.extend(
            self.fc2
                .named_parameters()
                .into_iter()
                .map(|p| p.with_prefix("fc2")),
        );
        named
    }
}
