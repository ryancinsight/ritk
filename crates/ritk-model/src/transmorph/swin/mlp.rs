//! Swin-block MLP, Coeus-native.
//!
//! A two-layer feed-forward network (`Linear â†’ GELU â†’ Linear`) applied along the
//! channel (last) axis of a `[B, D, H, W, C]` token volume. Built on
//! [`coeus_nn::Linear`] over [`coeus_autograd::Var`] with the exact GELU
//! activation; gradients flow to both linear layers through the autograd graph.

use coeus_autograd::{gelu, Parameter, Var};
use coeus_core::Backend;
use coeus_nn::module::Module;
use coeus_nn::Linear;
use coeus_ops::BackendOps;

/// Two-layer channel-wise MLP with a GELU nonlinearity.
#[derive(Clone)]
pub struct Mlp<B: Backend + BackendOps<f32> + Default> {
    fc1: Linear<f32, B>,
    fc2: Linear<f32, B>,
}

impl<B> Mlp<B>
where
    B: Backend + BackendOps<f32> + Default,
{
    /// Construct an MLP mapping `input_dim â†’ hidden_dim â†’ input_dim`.
    ///
    /// Weights are Kaiming-uniform-initialized (fan-in of each layer), biases
    /// zero â€” the non-degenerate scheme the original Burn model relied on;
    /// [`Linear::new`] alone leaves weights at ones.
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let mut fc1 = Linear::new(input_dim, hidden_dim, true);
        coeus_nn::init::kaiming_uniform_with_seed(&mut fc1.weight, input_dim, seed);
        let mut fc2 = Linear::new(hidden_dim, input_dim, true);
        coeus_nn::init::kaiming_uniform_with_seed(&mut fc2.weight, hidden_dim, seed ^ 0x5DEE_CE66);
        Self { fc1, fc2 }
    }

    /// Forward pass over a `[B, D, H, W, C]` token volume.
    pub fn forward(&self, x: &Var<f32, B>) -> Var<f32, B> {
        let x = self.fc1.forward(x);
        let x = gelu(&x);
        self.fc2.forward(&x)
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
