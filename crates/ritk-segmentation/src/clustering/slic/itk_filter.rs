use ritk_core::image::Image;
use ritk_image::tensor::{Backend, Tensor};
use ritk_tensor_ops::extract_vec_infallible;

use super::itk::slic_itk_impl;

/// Whether ITK SLIC perturbs initial centers to local gradient minima.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitializationPerturbation {
    /// Retain shrink-grid centers.
    Disabled,
    /// Move centers to local gradient minima.
    Enabled,
}

/// Whether ITK SLIC enforces face-connected output regions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConnectivityEnforcement {
    /// Retain disconnected label fragments.
    Disabled,
    /// Relabel disconnected fragments according to ITK semantics.
    Enabled,
}

/// Validated SimpleITK-compatible SLIC configuration.
#[derive(Clone, Debug)]
pub struct ItkSlicConfig {
    super_grid: usize,
    spatial_proximity_weight: f64,
    maximum_iterations: usize,
    perturbation: InitializationPerturbation,
    connectivity: ConnectivityEnforcement,
}

impl ItkSlicConfig {
    /// Create a fixed-grid SLIC configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when `super_grid` is zero.
    pub fn new(super_grid: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            super_grid >= 1,
            "ITK SLIC super-grid size must be at least 1, got {super_grid}"
        );
        Ok(Self {
            super_grid,
            spatial_proximity_weight: 10.0,
            maximum_iterations: 5,
            perturbation: InitializationPerturbation::Enabled,
            connectivity: ConnectivityEnforcement::Enabled,
        })
    }

    /// Set ITK's spatial proximity weight.
    pub fn with_spatial_proximity_weight(mut self, weight: f64) -> anyhow::Result<Self> {
        anyhow::ensure!(
            weight.is_finite() && weight >= 0.0 && weight <= (f64::MAX / 4.0).sqrt(),
            "ITK SLIC spatial proximity weight must be finite, nonnegative, and distance-representable, got {weight}"
        );
        self.spatial_proximity_weight = weight;
        Ok(self)
    }

    /// Set the fixed Lloyd iteration count.
    pub fn with_maximum_iterations(mut self, iterations: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            iterations >= 1,
            "ITK SLIC maximum iterations must be at least 1, got {iterations}"
        );
        self.maximum_iterations = iterations;
        Ok(self)
    }

    /// Set initialization perturbation policy.
    #[must_use]
    pub fn with_initialization_perturbation(
        mut self,
        perturbation: InitializationPerturbation,
    ) -> Self {
        self.perturbation = perturbation;
        self
    }

    /// Set connectivity enforcement policy.
    #[must_use]
    pub fn with_connectivity(mut self, connectivity: ConnectivityEnforcement) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Return the uniform super-grid size.
    pub fn super_grid(&self) -> usize {
        self.super_grid
    }

    /// Return ITK's spatial proximity weight.
    pub fn spatial_proximity_weight(&self) -> f64 {
        self.spatial_proximity_weight
    }

    /// Return the fixed Lloyd iteration count.
    pub fn maximum_iterations(&self) -> usize {
        self.maximum_iterations
    }

    /// Return the initialization perturbation policy.
    pub fn initialization_perturbation(&self) -> InitializationPerturbation {
        self.perturbation
    }

    /// Return the connectivity enforcement policy.
    pub fn connectivity(&self) -> ConnectivityEnforcement {
        self.connectivity
    }
}

/// SimpleITK-compatible fixed-grid SLIC filter.
pub struct ItkSlicFilter {
    config: ItkSlicConfig,
}

impl ItkSlicFilter {
    /// Create a filter from validated configuration.
    pub fn new(config: ItkSlicConfig) -> Self {
        Self { config }
    }

    /// Apply fixed-grid SLIC to a legacy 3-D image (`z == 1` is treated as 2-D).
    ///
    /// # Errors
    ///
    /// Returns an error for zero extents or non-finite samples.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (values, shape) = extract_vec_infallible(image);
        validate_input(&values, shape, self.config.super_grid)?;
        let labels = self.labels(&values, shape);
        let device = B::default();
        let tensor = Tensor::<f32, B>::from_slice_on(shape, &labels, &device);
        Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        )
    }

    /// Apply fixed-grid SLIC to a Coeus-native 3-D image.
    ///
    /// # Errors
    ///
    /// Returns an input-validation, backend storage, or output-construction error.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let values = image.data_slice()?;
        validate_input(values, image.shape(), self.config.super_grid)?;
        crate::native_output::from_values(image, self.labels(values, image.shape()), backend)
    }

    fn labels(&self, values: &[f32], [z, y, x]: [usize; 3]) -> Vec<f32> {
        if z == 1 {
            self.labels_for_shape(values, &[y, x], &[self.config.super_grid; 2])
        } else {
            self.labels_for_shape(values, &[z, y, x], &[self.config.super_grid; 3])
        }
    }

    fn labels_for_shape(&self, values: &[f32], shape: &[usize], grid: &[usize]) -> Vec<f32> {
        slic_itk_impl(
            values,
            shape,
            grid,
            self.config.spatial_proximity_weight,
            self.config.maximum_iterations,
            self.config.perturbation == InitializationPerturbation::Enabled,
            self.config.connectivity == ConnectivityEnforcement::Enabled,
        )
    }
}

fn validate_input(values: &[f32], shape: [usize; 3], super_grid: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        shape.iter().all(|&extent| extent > 0),
        "ITK SLIC requires nonzero dimensions, got {shape:?}"
    );
    let active_shape: &[usize] = if shape[0] == 1 { &shape[1..] } else { &shape };
    let label_count = active_shape.iter().try_fold(1usize, |count, &extent| {
        count.checked_mul((extent / super_grid).max(1))
    }).ok_or_else(|| anyhow::anyhow!("ITK SLIC label count overflows usize for shape {shape:?} and super-grid {super_grid}"))?;
    anyhow::ensure!(
        label_count <= super::MAX_EXACT_LABELS,
        "ITK SLIC label count must not exceed {} for exact f32 labels, got {label_count}",
        super::MAX_EXACT_LABELS
    );
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("ITK SLIC sample at flat index {index} must be finite, got {value}");
    }
    Ok(())
}
