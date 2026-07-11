//! Trainable Coeus displacement field and its validated geometry boundary.

use coeus_autograd::Var;
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::{BackendOps, Dimension, SupportedDimension};
use coeus_tensor::{StateDict, Tensor};
use ritk_core::spatial::{Direction, Point, Spacing};

/// Contract failures while constructing or loading a trainable field.
#[derive(Debug, thiserror::Error)]
pub enum DisplacementFieldError {
    /// Component count does not equal the spatial dimension.
    #[error("displacement component count mismatch: expected {expected}, got {actual}")]
    ComponentCount { expected: usize, actual: usize },
    /// A component shape differs from the canonical field shape.
    #[error(
        "displacement component {component} shape mismatch: expected {expected:?}, got {actual:?}"
    )]
    ComponentShape {
        component: usize,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// A component rank differs from the spatial dimension.
    #[error("displacement component rank mismatch: expected {expected}, got {actual}")]
    ComponentRank { expected: usize, actual: usize },
    /// The physical direction matrix is singular.
    #[error("displacement direction matrix is singular")]
    SingularDirection,
    /// A required named state tensor is absent.
    #[error("displacement state is missing tensor '{0}'")]
    MissingState(String),
    /// A geometry state tensor has an invalid shape.
    #[error("displacement state tensor '{name}' has shape {actual:?}, expected {expected:?}")]
    StateShape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// A persisted geometry value is non-finite or violates its domain.
    #[error("displacement state contains invalid {0} geometry")]
    InvalidGeometry(&'static str),
}

/// Dense trainable displacement vectors on a regular physical grid.
#[derive(Clone)]
pub struct DisplacementField<B: Backend, const D: usize>
where
    B: BackendOps<f32>,
{
    pub(crate) components: Vec<Var<f32, B>>,
    pub(crate) origin: Point<D>,
    pub(crate) spacing: Spacing<D>,
    pub(crate) direction: Direction<D>,
    pub(crate) world_to_index_matrix: Var<f32, B>,
    pub(crate) origin_tensor: Var<f32, B>,
}

impl<B: Backend + BackendOps<f32>, const D: usize> DisplacementField<B, D>
where
    Dimension<D>: SupportedDimension,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Validate components and construct a trainable field.
    pub fn new(
        components: Vec<Tensor<f32, B>>,
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> Result<Self, DisplacementFieldError> {
        if components.len() != D {
            return Err(DisplacementFieldError::ComponentCount {
                expected: D,
                actual: components.len(),
            });
        }
        let shape = components
            .first()
            .expect("invariant: supported dimensions are nonzero and component count equals D")
            .shape()
            .to_vec();
        if shape.len() != D {
            return Err(DisplacementFieldError::ComponentRank {
                expected: D,
                actual: shape.len(),
            });
        }
        for (component, tensor) in components.iter().enumerate().skip(1) {
            if tensor.shape() != shape {
                return Err(DisplacementFieldError::ComponentShape {
                    component,
                    expected: shape.clone(),
                    actual: tensor.shape().to_vec(),
                });
            }
        }

        let inverse = direction
            .try_inverse()
            .ok_or(DisplacementFieldError::SingularDirection)?;
        let origin_values = (0..D).map(|axis| origin[axis] as f32).collect::<Vec<_>>();
        let matrix_values = (0..D)
            .flat_map(|row| {
                (0..D).map(move |column| (inverse[(column, row)] / spacing[column]) as f32)
            })
            .collect::<Vec<_>>();
        let backend = B::default();

        Ok(Self {
            components: components
                .into_iter()
                .map(|component| Var::new(component, true))
                .collect(),
            origin,
            spacing,
            direction,
            world_to_index_matrix: Var::new(
                Tensor::from_slice_on([D, D], &matrix_values, &backend),
                false,
            ),
            origin_tensor: Var::new(
                Tensor::from_slice_on([1, D], &origin_values, &backend),
                false,
            ),
        })
    }

    /// Borrow the trainable component variables.
    #[must_use]
    pub fn components(&self) -> &[Var<f32, B>] {
        &self.components
    }

    /// Physical origin.
    #[must_use]
    pub const fn origin(&self) -> &Point<D> {
        &self.origin
    }

    /// Physical voxel spacing.
    #[must_use]
    pub const fn spacing(&self) -> &Spacing<D> {
        &self.spacing
    }

    /// Physical direction cosine matrix.
    #[must_use]
    pub const fn direction(&self) -> Direction<D> {
        self.direction
    }

    /// Materialize the complete named tensor state for bounded Coeus archival.
    #[must_use]
    pub fn state_dict(&self) -> StateDict<f32, B> {
        let backend = B::default();
        let mut state = StateDict::new();
        for (axis, component) in self.components.iter().enumerate() {
            state.insert(format!("field.component.{axis}"), component.tensor.clone());
        }
        state.insert(
            "field.origin",
            Tensor::from_slice_on(
                [D],
                &(0..D)
                    .map(|axis| self.origin[axis] as f32)
                    .collect::<Vec<_>>(),
                &backend,
            ),
        );
        state.insert(
            "field.spacing",
            Tensor::from_slice_on(
                [D],
                &(0..D)
                    .map(|axis| self.spacing[axis] as f32)
                    .collect::<Vec<_>>(),
                &backend,
            ),
        );
        state.insert(
            "field.direction",
            Tensor::from_slice_on(
                [D, D],
                &(0..D)
                    .flat_map(|row| (0..D).map(move |column| self.direction[(row, column)] as f32))
                    .collect::<Vec<_>>(),
                &backend,
            ),
        );
        state
    }

    /// Reconstruct a field from a validated materialized Coeus state archive.
    pub fn from_state_dict(state: &StateDict<f32, B>) -> Result<Self, DisplacementFieldError> {
        let origin = state_values(state, "field.origin", &[D])?;
        let spacing = state_values(state, "field.spacing", &[D])?;
        let direction = state_values(state, "field.direction", &[D, D])?;
        if origin.iter().any(|value| !value.is_finite()) {
            return Err(DisplacementFieldError::InvalidGeometry("origin"));
        }
        if spacing
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(DisplacementFieldError::InvalidGeometry("spacing"));
        }
        if direction.iter().any(|value| !value.is_finite()) {
            return Err(DisplacementFieldError::InvalidGeometry("direction"));
        }
        let components = (0..D)
            .map(|axis| {
                let name = format!("field.component.{axis}");
                state
                    .get(&name)
                    .cloned()
                    .ok_or(DisplacementFieldError::MissingState(name))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(
            components,
            Point::new(std::array::from_fn(|axis| origin[axis] as f64)),
            Spacing::new(std::array::from_fn(|axis| spacing[axis] as f64)),
            Direction::from_rows(std::array::from_fn(|row| {
                std::array::from_fn(|column| direction[row * D + column] as f64)
            })),
        )
    }
}

fn state_values<B: Backend + BackendOps<f32>>(
    state: &StateDict<f32, B>,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, DisplacementFieldError>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let tensor = state
        .get(name)
        .ok_or_else(|| DisplacementFieldError::MissingState(name.to_owned()))?;
    if tensor.shape() != expected {
        return Err(DisplacementFieldError::StateShape {
            name: name.to_owned(),
            expected: expected.to_vec(),
            actual: tensor.shape().to_vec(),
        });
    }
    Ok(tensor.to_contiguous().as_slice().to_vec())
}
