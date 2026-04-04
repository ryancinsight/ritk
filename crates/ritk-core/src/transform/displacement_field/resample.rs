use super::core::DisplacementField;
use crate::interpolation::{Interpolator, LinearInterpolator};
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

impl<B: Backend, const D: usize> DisplacementField<B, D> {
    /// # Theorem: Continuous Field Resampling
    /// Maps a pre-existing vector deformation field across newly defined rigid geometries.
    /// Let $F(x) \subset \mathbb{R}^D$ define an explicit functional vector map constraint, resampling formulates:
    /// $$ F'(y) = F( T^{-1}(y) ) $$
    ///
    /// By determining affine target positions dynamically matching local space intervals exactly, linear projection
    /// enforces mathematical equivalence mapping bounded geometric states precisely onto the local continuous components.
    pub fn resample(
        &self,
        new_shape: [usize; D],
        new_origin: Point<D>,
        new_spacing: Spacing<D>,
        new_direction: Direction<D>,
    ) -> Self {
        let device = self.components[0].device();
        let interpolator = LinearInterpolator::new();

        let new_indices = crate::image::grid::generate_grid(new_shape, &device);

        let mut m_data = Vec::with_capacity(D * D);
        for k in 0..D {
            for c in 0..D {
                let val = (new_direction[(c, k)] * new_spacing[k]) as f32;
                m_data.push(val);
            }
        }
        let m_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(m_data, Shape::new([D, D])), &device);

        let origin_vec: Vec<f32> = (0..D).map(|i| new_origin[i] as f32).collect();
        let origin_tensor =
            Tensor::<B, 1>::from_data(TensorData::new(origin_vec, Shape::new([D])), &device)
                .reshape([1, D]);

        let [n_points, _] = new_indices.dims();
        const CHUNK_SIZE: usize = 32768;

        let mut component_chunks: Vec<Vec<Tensor<B, 1>>> = vec![Vec::new(); D];
        let num_chunks = (n_points + CHUNK_SIZE - 1) / CHUNK_SIZE;

        for i in 0..num_chunks {
            let start = i * CHUNK_SIZE;
            let end = std::cmp::min(start + CHUNK_SIZE, n_points);
            let chunk_indices = new_indices.clone().slice([start..end]);

            let offset = chunk_indices.matmul(m_tensor.clone());
            let world = offset + origin_tensor.clone();

            let old_indices = self.world_to_index_tensor(world);

            for d in 0..D {
                let comp = &self.components[d].val();
                let val = interpolator.interpolate(comp, old_indices.clone());
                component_chunks[d].push(val);
            }
        }

        let mut final_components = Vec::with_capacity(D);
        for d in 0..D {
            let flat = Tensor::cat(component_chunks[d].clone(), 0);
            let reshaped = flat.reshape(Shape::new(new_shape));
            final_components.push(reshaped);
        }

        Self::new(final_components, new_origin, new_spacing, new_direction)
    }
}
