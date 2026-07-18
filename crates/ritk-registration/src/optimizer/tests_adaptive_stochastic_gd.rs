use super::*;
use burn_ndarray::NdArray;
use ritk_image::burn::backend::Autodiff;
use ritk_image::burn::module::Module;
use ritk_image::tensor::Tensor;
use ritk_image::tensor::{Backend };

type TestBackend = Autodiff<NdArray<f32>>;

#[derive(Module, Debug)]
struct Quadratic<B: Backend> {
    x: Param<Tensor<f32, B>> }

impl<B: Backend> Quadratic<B> {
    fn new(x0: &[f32], device: &B::Device) -> Self {
        let x = Tensor::<f32, B>::from_data(::from(x0), device);
        Self {
            x: Param::from_tensor(x) }
    }

    fn forward(&self) -> Tensor<f32, B> {
        let x = self.x.val();
        x.clone() * x
    }

    fn loss_value(&self) -> f64 {
        let x = self.x.val();
        let data = x.to_data();
        let slice = data
            .as_slice::<f32>()
            .expect("gradient tensor data must be contiguous f32");
        slice.iter().map(|&v| (v as f64) * (v as f64)).sum()
    }
}

#[test]
fn asgd_minimizes_quadratic_function() {
    let device = Default::default();
    let mut module = Quadratic::<TestBackend>::new(&[5.0, -3.0], &device);

    let config = AdaptiveStochasticGdConfig {
        a: 0.1,
        a_damping: 0.0,
        alpha: 0.5,
        sigmoid_max: 1.0,
        sigmoid_min: -0.5,
        sigmoid_scale: 1e-4,
        gradient_tolerance: 1e-6,
        maximum_iterations: 500 };

    let mut optimizer: AdaptiveStochasticGradientDescent<Quadratic<TestBackend>, TestBackend> =
        AdaptiveStochasticGradientDescent::new(config);

    let initial_loss = module.loss_value();

    for _ in 0..500 {
        if optimizer.converged() {
            break;
        }
        let loss = module.forward();
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);
        module = optimizer.step(module, grads_params);
    }

    let final_loss = module.loss_value();
    assert!(final_loss < initial_loss * 0.1, "ASGD failed to minimize");
}

#[test]
fn asgd_config_validation() {
    let mut cfg = AdaptiveStochasticGdConfig::default();
    assert!(cfg.validate().is_ok());

    cfg.a = -1.0;
    assert!(cfg.validate().is_err());

    cfg = AdaptiveStochasticGdConfig::default();
    cfg.sigmoid_max = -1.0; // <= sigmoid_min
    assert!(cfg.validate().is_err());
}
