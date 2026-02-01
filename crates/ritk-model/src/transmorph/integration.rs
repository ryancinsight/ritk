use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use super::spatial_transform::SpatialTransformer;

/// Velocity Integration Module (Scaling and Squaring).
///
/// Integrates a stationary velocity field to produce a diffeomorphic
/// displacement field.
/// 
/// $\phi = \exp(v)$
///
/// Implemented via scaling and squaring:
/// 1. Scale flow by $1/2^N$
/// 2. Recursively compose $v_{i+1} = v_i + v_i \circ (x + v_i)$ for N steps.
#[derive(Module, Debug)]
pub struct VecInt<B: Backend> {
    stn: SpatialTransformer<B>,
    nsteps: usize,
}

impl<B: Backend> VecInt<B> {
    /// Create a new Velocity Integrator.
    ///
    /// # Arguments
    /// * `nsteps` - Number of integration steps (e.g., 7 for 128 steps)
    pub fn new(nsteps: usize) -> Self {
        Self {
            stn: SpatialTransformer::new(),
            nsteps,
        }
    }

    /// Forward pass: Integrate velocity field.
    ///
    /// # Arguments
    /// * `flow` - Velocity field [B, 3, D, H, W]
    ///
    /// # Returns
    /// * `displacement` - Integrated displacement field [B, 3, D, H, W]
    pub fn forward(&self, flow: Tensor<B, 5>) -> Tensor<B, 5> {
        // flow: [B, 3, D, H, W]
        // Scale flow
        let scale = 1.0 / (2.0f32).powi(self.nsteps as i32);
        let mut flow = flow * scale;
        
        // Squaring
        for _ in 0..self.nsteps {
            // flow = flow + flow \circ (Id + flow)
            // STN computes image(x + flow).
            // Here "image" is "flow".
            // So STN(flow, flow) = flow(x + flow(x))
            // This is exactly the composition term needed.
            let warped_flow = self.stn.forward(flow.clone(), flow.clone());
            flow = flow + warped_flow;
        }
        
        flow
    }
}
