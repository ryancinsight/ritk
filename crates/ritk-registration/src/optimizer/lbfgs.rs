//! L-BFGS Optimizer Implementation
//!
//! Limited-memory BFGS optimization algorithm for registration.
//! This implementation follows the standard L-BFGS two-loop recursion algorithm.

use crate::optimizer::Optimizer;
use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, SimpleOptimizer, Optimizer as BurnOptimizer};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::tensor::{Tensor, ElementConversion};
use burn::tensor::backend::{Backend, AutodiffBackend};
use burn::config::Config;
use burn::record::Record;
use std::marker::PhantomData;

/// L-BFGS Configuration
#[derive(Config, Debug)]
pub struct LbfgsConfig {
    /// History size (number of steps to keep)
    #[config(default = 10)]
    pub history_size: usize,
    /// Learning rate (step size)
    #[config(default = 1.0)]
    pub learning_rate: f64,
}

impl LbfgsConfig {
    /// Initialize L-BFGS optimizer
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(&self) -> LbfgsOptimizer<M, B> {
        LbfgsOptimizer {
            optimizer: OptimizerAdaptor::from(LbfgsCore::new(self.history_size, self.learning_rate)),
            learning_rate: self.learning_rate,
        }
    }
}

/// L-BFGS Optimizer wrapper
pub struct LbfgsOptimizer<M: AutodiffModule<B>, B: AutodiffBackend> {
    optimizer: OptimizerAdaptor<LbfgsCore<B::InnerBackend>, M, B>,
    learning_rate: f64,
}

impl<M: AutodiffModule<B>, B: AutodiffBackend> LbfgsOptimizer<M, B> {
    /// Create a new L-BFGS optimizer with default config
    pub fn new(learning_rate: f64) -> Self {
        LbfgsConfig::new()
            .with_learning_rate(learning_rate)
            .init()
    }

    /// Create with history size
    pub fn with_history(mut self, history_size: usize) -> Self {
        self.optimizer = OptimizerAdaptor::from(LbfgsCore::new(history_size, self.learning_rate));
        self
    }
}

impl<M, B> Optimizer<M, B> for LbfgsOptimizer<M, B>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    fn step(&mut self, module: M, gradients: GradientsParams) -> M {
        self.optimizer.step(self.learning_rate, module, gradients)
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}

/// L-BFGS Core Implementation
#[derive(Clone, Debug)]
struct LbfgsCore<B: Backend> {
    history_size: usize,
    _learning_rate: f64,
    _phantom: PhantomData<B>,
}

impl<B: Backend> LbfgsCore<B> {
    pub fn new(history_size: usize, learning_rate: f64) -> Self {
        Self {
            history_size,
            _learning_rate: learning_rate,
            _phantom: PhantomData,
        }
    }
}

/// L-BFGS State
#[derive(Clone, Debug, Record)]
pub struct LbfgsState<B: Backend> {
    /// History of s (parameter differences: x_k - x_{k-1})
    pub s_history: Vec<Tensor<B, 1>>,
    /// History of y (gradient differences: g_k - g_{k-1})
    pub y_history: Vec<Tensor<B, 1>>,
    /// History of rho (1 / y^T s)
    pub rho_history: Vec<Tensor<B, 1>>,
    /// Previous flattened parameter value
    pub prev_param: Option<Tensor<B, 1>>,
    /// Previous flattened gradient value
    pub prev_grad: Option<Tensor<B, 1>>,
}

/// State type alias to work around associated type issues
type State<B> = LbfgsState<B>;

impl<B: Backend> SimpleOptimizer<B> for LbfgsCore<B> {
    type State<const D: usize> = State<B>;

    fn step<const D: usize>(
        &self,
        lr: f64,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let shape = tensor.shape();
        let _device = tensor.device();
        let dim_size = shape.num_elements();

        // Flatten to 1D for L-BFGS algorithm
        let t1 = tensor.reshape([dim_size]);
        let g1 = grad.reshape([dim_size]);

        let (new_t1, new_state) = match state {
            None => {
                // First iteration: gradient descent step
                let update = g1.clone().mul_scalar(lr);
                let new_t1 = t1.clone().sub(update);

                let state = LbfgsState {
                    s_history: Vec::new(),
                    y_history: Vec::new(),
                    rho_history: Vec::new(),
                    prev_param: Some(t1),
                    prev_grad: Some(g1),
                };

                (new_t1, Some(state))
            }
            Some(mut state) => {
                let prev_param = state.prev_param.as_ref().expect("prev_param should exist");
                let prev_grad = state.prev_grad.as_ref().expect("prev_grad should exist");

                // Compute s_k = x_k - x_{k-1} and y_k = g_k - g_{k-1}
                let s_k = t1.clone().sub(prev_param.clone());
                let y_k = g1.clone().sub(prev_grad.clone());

                // Compute rho_k = 1 / (y_k^T * s_k)
                let y_dot_s = y_k.clone().mul(s_k.clone()).sum();
                let ys_scalar = y_dot_s.clone().into_scalar().elem::<f64>();

                // Update history if curvature condition is satisfied (y^T s > epsilon)
                if ys_scalar > 1e-10 {
                    let rho_k = y_dot_s.recip();

                    // Maintain fixed-size queue
                    if state.s_history.len() >= self.history_size {
                        state.s_history.remove(0);
                        state.y_history.remove(0);
                        state.rho_history.remove(0);
                    }

                    state.s_history.push(s_k);
                    state.y_history.push(y_k);
                    state.rho_history.push(rho_k);
                }

                // L-BFGS Two-Loop Recursion
                let q = g1.clone();
                let mut alphas = Vec::new();

                // First loop (backward)
                let len = state.s_history.len();
                let mut q_curr = q;

                for i in (0..len).rev() {
                    let s_i = &state.s_history[i];
                    let y_i = &state.y_history[i];
                    let rho_i = &state.rho_history[i];

                    let alpha = rho_i.clone().mul(s_i.clone().mul(q_curr.clone()).sum());
                    alphas.push(alpha.clone());

                    q_curr = q_curr.sub(y_i.clone().mul(alpha));
                }

                // Apply initial Hessian approximation
                let r = if len > 0 {
                    let last_s = &state.s_history[len - 1];
                    let last_y = &state.y_history[len - 1];

                    // gamma = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
                    let num = last_s.clone().mul(last_y.clone()).sum();
                    let den = last_y.clone().mul(last_y.clone()).sum();
                    let gamma = num.div(den);

                    q_curr.mul(gamma)
                } else {
                    q_curr.mul_scalar(lr)
                };

                // Second loop (forward)
                let mut r_curr = r;
                for i in 0..len {
                    let s_i = &state.s_history[i];
                    let y_i = &state.y_history[i];
                    let rho_i = &state.rho_history[i];

                    // alphas were collected in reverse order
                    let alpha = &alphas[len - 1 - i];

                    let beta = rho_i.clone().mul(y_i.clone().mul(r_curr.clone()).sum());
                    let term = s_i.clone().mul(alpha.clone().sub(beta));

                    r_curr = r_curr.add(term);
                }

                // Update parameters: x_{k+1} = x_k - lr * direction
                // r_curr is H^{-1} * g, so stepping in direction -r_curr
                let step = r_curr.mul_scalar(lr);
                let new_t1 = t1.clone().sub(step);

                // Update state for next iteration
                state.prev_param = Some(t1);
                state.prev_grad = Some(g1);

                (new_t1, Some(state))
            }
        };

        // Reshape back to original shape
        let new_tensor = new_t1.reshape(shape);

        (new_tensor, new_state)
    }

    fn to_device<const D: usize>(
        state: Self::State<D>,
        device: &<B>::Device,
    ) -> Self::State<D> {
        LbfgsState {
            s_history: state.s_history
                .into_iter()
                .map(|t| t.to_device(device))
                .collect(),
            y_history: state.y_history
                .into_iter()
                .map(|t| t.to_device(device))
                .collect(),
            rho_history: state.rho_history
                .into_iter()
                .map(|t| t.to_device(device))
                .collect(),
            prev_param: state.prev_param.map(|t| t.to_device(device)),
            prev_grad: state.prev_grad.map(|t| t.to_device(device)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::module::{Module, Param};
    use burn::tensor::Data;
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray<f32>>;

    #[derive(Module, Debug)]
    struct LinearModel<B: Backend> {
        weight: Param<Tensor<B, 1>>,
    }

    impl<B: Backend> LinearModel<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                weight: Param::from_tensor(Tensor::from_floats([0.0], device).require_grad()),
            }
        }

        fn forward(&self) -> Tensor<B, 1> {
            self.weight.val()
        }
    }

    #[test]
    fn test_lbfgs_quadratic() {
        let device = Default::default();
        let mut model = LinearModel::<TestBackend>::new(&device);
        let mut optimizer = LbfgsOptimizer::new(1.0);

        let target = Tensor::<TestBackend, 1>::from_floats([5.0], &device);

        for i in 0..20 {
            let pred = model.forward();
            let loss = pred.sub(target.clone()).powf_scalar(2.0).mean();

            if i % 5 == 0 {
                let loss_val = loss.clone().into_scalar().elem::<f32>();
                println!("Iteration {}: loss = {:.6}", i, loss_val);
            }

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(model, grads_params);
        }

        let final_pred = model.forward().into_scalar().elem::<f32>();
        assert!((final_pred - 5.0).abs() < 0.1, "L-BFGS should optimize to target");
    }
}
