//! Swin feed-forward projection.

use coeus_autograd::Var;
use coeus_core::Backend;
use coeus_nn::{Dropout, Linear, Module};
use coeus_ops::BackendOps;

/// Two-layer GELU feed-forward network applied along the final axis.
#[derive(Clone)]
pub struct Mlp<B>
where
    B: Backend + BackendOps<f32>,
{
    first: Linear<f32, B>,
    second: Linear<f32, B>,
    dropout: Dropout,
}

/// Feed-forward network configuration.
#[derive(Debug, Clone, Copy)]
pub struct MlpConfig {
    input_dim: usize,
    hidden_dim: usize,
    dropout: f64,
}

impl MlpConfig {
    /// Construct a feed-forward configuration.
    #[must_use]
    pub const fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            dropout: 0.0,
        }
    }

    /// Set the dropout probability.
    #[must_use]
    pub const fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Initialize the network on backend `B`.
    #[must_use]
    pub fn init<B>(self) -> Mlp<B>
    where
        B: Backend + BackendOps<f32>,
    {
        let mut mlp = Mlp {
            first: Linear::new(self.input_dim, self.hidden_dim, true),
            second: Linear::new(self.hidden_dim, self.input_dim, true),
            dropout: Dropout::new(self.dropout),
        };
        crate::initialization::linear(&mut mlp.first, self.input_dim, self.hidden_dim, 201);
        crate::initialization::linear(&mut mlp.second, self.hidden_dim, self.input_dim, 202);
        mlp
    }
}

impl<B> Module<f32, B> for Mlp<B>
where
    B: Backend + BackendOps<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut parameters = self.first.parameters();
        parameters.extend(self.second.parameters());
        parameters
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        let hidden = self.first.forward(input);
        let hidden = coeus_autograd::gelu(&hidden);
        let hidden = self.dropout.forward(&hidden);
        let output = self.second.forward(&hidden);
        self.dropout.forward(&output)
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        let first_len = self.first.parameters().len();
        self.first.load_parameters(&parameters[..first_len]);
        self.second.load_parameters(&parameters[first_len..]);
    }

    fn train(&mut self, mode: bool) {
        self.dropout.set_training(mode);
    }
}
