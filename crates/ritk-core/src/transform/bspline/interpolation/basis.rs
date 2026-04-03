use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use crate::transform::bspline::BSplineTransform;

impl<B: Backend, const D: usize> BSplineTransform<B, D> {
    /// Compute Cubic B-Spline basis functions.
    pub(crate) fn bspline_basis(u: Tensor<B, 1>) -> [Tensor<B, 1>; 4] {
        // u is in [0, 1)
        let one = 1.0;
        let two = 2.0;
        let three = 3.0;
        let four = 4.0;
        let six = 6.0;

        // B0 = (1-u)^3 / 6
        let one_minus_u = u.clone().neg().add_scalar(one);
        let b0 = one_minus_u.powf_scalar(three) / six;

        // B1 = (3u^3 - 6u^2 + 4) / 6
        let u2 = u.clone().powf_scalar(two);
        let u3 = u.clone().powf_scalar(three);
        let b1 = (u3.clone().mul_scalar(three) - u2.clone().mul_scalar(six)).add_scalar(four) / six;

        // B2 = (-3u^3 + 3u^2 + 3u + 1) / 6
        let b2 = (u3.clone().mul_scalar(-three)
            + u2.clone().mul_scalar(three)
            + u.clone().mul_scalar(three))
        .add_scalar(one)
            / six;

        // B3 = u^3 / 6
        let b3 = u3 / six;

        [b0, b1, b2, b3]
    }

    /// Compute Cubic B-Spline basis functions and stack them into a tensor [Batch, 4].
    pub(crate) fn compute_basis_tensor(u: Tensor<B, 1>) -> Tensor<B, 2> {
        let [b0, b1, b2, b3] = Self::bspline_basis(u);
        // Stack along dim 1: [Batch, 1] -> [Batch, 4]
        Tensor::cat(
            vec![
                b0.unsqueeze_dim::<2>(1),
                b1.unsqueeze_dim::<2>(1),
                b2.unsqueeze_dim::<2>(1),
                b3.unsqueeze_dim::<2>(1),
            ],
            1,
        )
    }
}
