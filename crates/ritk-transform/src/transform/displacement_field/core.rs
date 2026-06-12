//! Displacement field core data structure definition and invariant construction.
//!
//! Exposes continuous field projections mathematically validated against geometric origins.

use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::spatial::{Direction, Point, Spacing};

/// Displacement field data representing a dense vector field on a regular mathematical grid.
///
/// Maintains vector coordinates mapping directly over strict geometric domains enabling mathematically verifiable transformations.
#[derive(Module, Debug)]
pub struct DisplacementField<B: Backend, const D: usize> {
    pub(crate) components: Vec<Param<Tensor<B, D>>>,
    pub(crate) origin: Point<D>,
    pub(crate) spacing: Spacing<D>,
    pub(crate) direction: Direction<D>,

    pub(crate) world_to_index_matrix: Tensor<B, 2>,
    pub(crate) origin_tensor: Tensor<B, 2>,
}

impl<B: Backend, const D: usize> DisplacementField<B, D> {
    /// # Theorem: Discrete-to-Continuous Coordinate Projection
    /// The construction of the index mapping matrix establishes an invertible affine transformation
    /// between discrete tensor voxel coordinates and the continuous physical domain $\mathbb{R}^D$.
    ///
    /// Let $S$ be the diagonal matrix of physical spacing and $D_{ir}$ be the orthogonal
    /// direction cosine matrix. The transformation from index space $v$ to world coordinates $w$
    /// is geometrically proven as:
    /// $$ w = v^T (S \cdot D_{ir}) + O $$
    ///
    /// The inverse continuous mapping constructing the dense coordinate matrix isolates constants via:
    /// $$ v^T = (w - O) (S \cdot D_{ir})^{-1} = (w - O) (S^{-1} D_{ir}^{-1})^T $$
    ///
    /// Where the matrix $T = (S^{-1} D_{ir}^{-1})^T$ rigorously isolates operations allowing parallel batched execution
    /// over infinite continuous dimensions mapping without explicit reallocations.
    pub fn new(
        components: Vec<Tensor<B, D>>,
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> Self {
        assert_eq!(
            components.len(),
            D,
            "Number of components must match dimensionality"
        );
        if !components.is_empty() {
            let shape = components[0].shape();
            for c in &components[1..] {
                assert_eq!(
                    c.shape(),
                    shape,
                    "All components must have identical domains"
                );
            }
        }

        let device = components[0].device();

        let components = components
            .into_iter()
            .map(|c| Param::from_tensor(c.require_grad()))
            .collect();

        let origin_vec: Vec<f32> = (0..D).map(|i| origin[i] as f32).collect();
        let origin_tensor =
            Tensor::<B, 1>::from_data(TensorData::new(origin_vec, Shape::new([D])), &device)
                .reshape([1, D]);

        let inv_dir = direction
            .try_inverse()
            .expect("Direction matrix mathematically non-invertible");

        let mut t_data = Vec::with_capacity(D * D);
        for r in 0..D {
            for c in 0..D {
                let val = (inv_dir[(c, r)] / spacing[c]) as f32;
                t_data.push(val);
            }
        }

        let world_to_index_matrix =
            Tensor::<B, 2>::from_data(TensorData::new(t_data, Shape::new([D, D])), &device);

        Self {
            components,
            origin,
            spacing,
            direction,
            world_to_index_matrix,
            origin_tensor,
        }
    }

    pub fn components(&self) -> Vec<Tensor<B, D>> {
        self.components.iter().map(|p| p.val()).collect()
    }

    pub fn origin(&self) -> &Point<D> {
        &self.origin
    }

    pub fn spacing(&self) -> &Spacing<D> {
        &self.spacing
    }

    pub fn direction(&self) -> Direction<D> {
        self.direction
    }
}

// Dimension-specific convenience aliases removed.
// Use `DisplacementField<B, 2>` or `DisplacementField<B, 3>` directly.
