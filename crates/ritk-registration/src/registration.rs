use burn::tensor::backend::AutodiffBackend;
use burn::module::AutodiffModule;
use burn::optim::GradientsParams;
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use crate::metric::Metric;
use crate::optimizer::Optimizer;
use std::marker::PhantomData;

pub struct Registration<B, O, M, T, const D: usize>
where
    B: AutodiffBackend,
    O: Optimizer<T, B>,
    M: Metric<B, D>,
    T: Transform<B, D> + AutodiffModule<B>,
{
    optimizer: O,
    metric: M,
    _phantom: PhantomData<(B, T)>,
}

impl<B, O, M, T, const D: usize> Registration<B, O, M, T, D>
where
    B: AutodiffBackend,
    O: Optimizer<T, B>,
    M: Metric<B, D>,
    T: Transform<B, D> + AutodiffModule<B>,
{
    pub fn new(optimizer: O, metric: M) -> Self {
        Self {
            optimizer,
            metric,
            _phantom: PhantomData,
        }
    }

    pub fn execute(
        &mut self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        mut transform: T,
        iterations: usize,
        learning_rate: f64,
    ) -> T {
        self.optimizer.set_learning_rate(learning_rate);
        
        for i in 0..iterations {
            // Forward pass
            let loss = self.metric.forward(fixed, moving, &transform);
            
            // Log loss periodically
            if i % 50 == 0 {
                let data = loss.to_data();
                let val = data.as_slice::<f32>().unwrap()[0];
                tracing::info!("Iteration {}: Loss {:.6}", i, val);
            }

            // Backward pass
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &transform);
            
            // Optimizer step
            transform = self.optimizer.step(transform, grads_params);
        }
        transform
    }
}
