use burn::{
    nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig},
    prelude::*,
    tensor::backend::Backend,
    tensor::Tensor,
};

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    act: Gelu,
    fc2: Linear<B>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct MlpConfig {
    input_dim: usize,
    hidden_dim: usize,
    #[config(default = 0.0)]
    dropout: f64,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        Mlp {
            fc1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            act: Gelu::new(),
            fc2: LinearConfig::new(self.hidden_dim, self.input_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let x = self.fc1.forward(x);
        let x = self.act.forward(x);
        let x = self.dropout.forward(x);
        let x = self.fc2.forward(x);
        self.dropout.forward(x)
    }
}
