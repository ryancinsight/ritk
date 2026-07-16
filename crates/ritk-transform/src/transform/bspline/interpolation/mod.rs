use super::BSplineTransform;
use coeus_core::CpuAddressableStorage;
use ritk_core::transform::Transform;
use coeus_core::Backend;
use coeus_tensor::Tensor;

fn cubic_bspline_basis(u: f32) -> [f32; 4] {
    let u2 = u * u;
    let u3 = u2 * u;
    [
        (1.0 - u).powi(3) / 6.0,
        (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0,
        (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0,
        u3 / 6.0,
    ]
}

impl<B: Backend, const D: usize> BSplineTransform<B, D> {
    const _SUPPORTED_DIM: () = assert!(
        matches!(D, 1..=4),
        "BSplineTransform only supports D ∈ {{1, 2, 3, 4}}"
    );
}

impl<B: Backend, const D: usize> Transform<B, D> for BSplineTransform<B, D>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    #[inline]
    fn transform_points(&self, points: Tensor<f32, B>) -> Tensor<f32, B> {
        let _: () = Self::_SUPPORTED_DIM;
        let device = B::default();
        let points = points.to_contiguous();
        let grid_coords = self.world_to_grid_tensor(points.clone()).to_contiguous();
        let coefficients = self.coefficients.to_contiguous();

        let batch = points.shape()[0];
        let point_data = points.as_slice();
        let grid_data = grid_coords.as_slice();
        let coeff_data = coefficients.as_slice();
        let mut output = point_data.to_vec();

        let mut strides = [1usize; D];
        for dim in 1..D {
            strides[dim] = strides[dim - 1] * self.grid_size[dim - 1];
        }

        for row in 0..batch {
            let mut coords = [0.0f32; D];
            let mut base = [0isize; D];
            let mut basis = [[0.0f32; 4]; D];
            let mut valid = true;

            for dim in 0..D {
                let value = grid_data[row * D + dim];
                if !(0.0..=(self.grid_size[dim] as f32 - 1.0)).contains(&value) {
                    valid = false;
                    break;
                }
                coords[dim] = value;
                let floored = value.floor();
                base[dim] = floored as isize - 1;
                basis[dim] = cubic_bspline_basis(value - floored);
            }

            if !valid {
                continue;
            }

            let support = 4usize.pow(D as u32);
            let mut displacement = [0.0f32; D];

            for linear in 0..support {
                let mut rem = linear;
                let mut weight = 1.0f32;
                let mut flat_index = 0usize;

                for dim in 0..D {
                    let offset = rem % 4;
                    rem /= 4;
                    weight *= basis[dim][offset];
                    let clamped = (base[dim] + offset as isize)
                        .clamp(0, self.grid_size[dim] as isize - 1) as usize;
                    flat_index += clamped * strides[dim];
                }

                let coeff_base = flat_index * D;
                for dim in 0..D {
                    displacement[dim] += weight * coeff_data[coeff_base + dim];
                }
            }

            for dim in 0..D {
                output[row * D + dim] += displacement[dim];
            }
        }

        Tensor::<f32, B>::from_slice_on([batch, D], &output, &device)
    }
}
