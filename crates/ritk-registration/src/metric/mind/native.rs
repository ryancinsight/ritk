//! Fixed-domain, on-demand MIND-SSC evaluation on native images.

use std::mem::{size_of, size_of_val};

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use eunomia::CastFrom;
use ritk_image::Image;

use crate::classical::rigid_physical_affine_to_native;
use crate::types::AffineTransform;

use super::config::DescriptorGeometry;
use super::descriptor::descriptor_at;
use super::geometry::{
    apply_native_affine, trilinear_background, validate_transform, CartesianGeometry,
};
use super::sampling::{checked_voxel_count, decode_index, linear_index, select_indices};
use super::{MindSscConfig, MindSscError};

/// Exact heap-payload accounting for prepared MIND-SSC fixed-domain state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct MindSscMemoryUsage {
    /// Number of fixed centers retained by the sampling policy.
    pub selected_centers: usize,
    /// Bytes occupied by retained linear center indices.
    pub index_bytes: usize,
    /// Bytes occupied by packed 60-bit descriptors in `u64` words.
    pub descriptor_bytes: usize,
    /// Bytes occupied by retained fixed weights.
    pub weight_bytes: usize,
    /// Sum of index, descriptor, and weight heap payloads.
    pub heap_payload_bytes: usize,
    /// Algorithmic scalar scratch per center: six patch values and 12 sums.
    pub per_center_scratch_bytes: usize,
}

/// Prepared fixed-domain MIND-SSC state for repeated rigid-pose evaluation.
///
/// Only selected center indices, one packed `u64` descriptor per center, and
/// optional selected weights persist. Each pose maps the six fixed-grid patch
/// neighbourhoods through the physical transform and samples the moving image
/// directly. No dense moving descriptor or resampled volume is constructed.
/// The selected center set and normalization denominator never depend on pose.
///
/// # Example
///
/// ```
/// use coeus_core::SequentialBackend;
/// use ritk_image::Image;
/// use ritk_registration::metric::mind::{MindSscConfig, MindSscFixedPrep};
/// use ritk_registration::types::AffineTransform;
/// use ritk_spatial::{Direction, Point, Spacing};
///
/// let image = Image::from_flat_on(
///     vec![4.0_f32; 7 * 7 * 7],
///     [7, 7, 7],
///     Point::origin(),
///     Spacing::uniform(1.0),
///     Direction::identity(),
///     &SequentialBackend,
/// )?;
/// let prepared = MindSscFixedPrep::try_new(&image, MindSscConfig::default(), None, None)?;
/// let score = prepared.eval(&image, &AffineTransform::IDENTITY)?;
/// assert_eq!(score, 1.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct MindSscFixedPrep<B>
where
    B: ComputeBackend,
{
    geometry: DescriptorGeometry,
    shape: [usize; 3],
    fixed_geometry: CartesianGeometry,
    indices: Box<[usize]>,
    descriptors: Box<[u64]>,
    weights: Option<Box<[f32]>>,
    denominator: f32,
    _backend: std::marker::PhantomData<B>,
}

impl<B> MindSscFixedPrep<B>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    /// Prepare selected complete-support descriptors on the fixed image.
    ///
    /// `mask` and `weights` use fixed-image C-order. A caller-index sampling
    /// policy requires every index to have complete support and pass the mask.
    ///
    /// # Errors
    ///
    /// Returns [`MindSscError`] for invalid image geometry or storage, malformed
    /// masks or weights, empty support, invalid caller indices, and non-finite
    /// fixed samples used by selected descriptors.
    pub fn try_new(
        fixed: &Image<f32, B, 3>,
        config: MindSscConfig,
        mask: Option<&[bool]>,
        weights: Option<&[f32]>,
    ) -> Result<Self, MindSscError> {
        let fixed_geometry = CartesianGeometry::try_from_image(fixed, "fixed")?;
        let fixed_values =
            fixed
                .data_slice()
                .map_err(|error| MindSscError::NonContiguousImage {
                    image: "fixed",
                    reason: error.to_string(),
                })?;
        let shape = fixed.shape();
        let voxel_count = checked_voxel_count(shape)?;
        if let Some(weights) = weights {
            if weights.len() != voxel_count {
                return Err(MindSscError::WeightLength {
                    expected: voxel_count,
                    actual: weights.len(),
                });
            }
        }
        let geometry = config.geometry();
        let spacing = [fixed.spacing()[2], fixed.spacing()[1], fixed.spacing()[0]];
        let indices = select_indices(config.sampling(), shape, config.halo(), spacing, mask)?;
        let mut descriptors = Vec::with_capacity(indices.len());
        let mut selected_weights = weights.map(|_| Vec::with_capacity(indices.len()));
        let mut weight_sum = 0.0_f32;
        for &center_index in &indices {
            let center = decode_index(center_index, shape)?;
            descriptors.push(descriptor_at(center, geometry, |support| {
                fixed_value(fixed_values, shape, support)
            })?);
            let weight: f32 = weights.map_or(1.0, |values| values[center_index]);
            if !weight.is_finite() || weight < 0.0 {
                return Err(MindSscError::InvalidWeight {
                    index: center_index,
                    value: weight,
                });
            }
            weight_sum += weight;
            if let Some(selected) = selected_weights.as_mut() {
                selected.push(weight);
            }
        }
        if weight_sum == 0.0 {
            return Err(MindSscError::ZeroSelectedWeight);
        }
        if !weight_sum.is_finite() {
            return Err(MindSscError::NonFiniteSelectedWeightSum { value: weight_sum });
        }
        let denominator = weight_sum * 60.0;
        if !denominator.is_finite() {
            return Err(MindSscError::NonFiniteDenominator { weight_sum });
        }
        Ok(Self {
            geometry,
            shape,
            fixed_geometry,
            indices: indices.into_boxed_slice(),
            descriptors: descriptors.into_boxed_slice(),
            weights: selected_weights.map(Vec::into_boxed_slice),
            denominator,
            _backend: std::marker::PhantomData,
        })
    }

    /// Evaluate a classical fixed-to-moving physical rigid affine.
    ///
    /// The affine follows [`crate::classical::search_rigid_pose`]: row-major
    /// physical `[z, y, x]` millimetres. RITK performs the axis bridge before
    /// sampling native `[x, y, z]` world points. Moving support outside ITK's
    /// half-voxel interval `[-0.5, size - 0.5)` on any continuous index axis
    /// contributes explicit zero background but remains in the fixed
    /// denominator. Samples inside that interval use replicated border values.
    ///
    /// # Errors
    ///
    /// Returns [`MindSscError`] when moving geometry/storage, affine conversion,
    /// sampled values, or descriptor arithmetic is invalid.
    pub fn eval(
        &self,
        moving: &Image<f32, B, 3>,
        transform: &AffineTransform,
    ) -> Result<f32, MindSscError> {
        let moving_shape = moving.shape();
        if moving_shape.contains(&0) {
            return Err(MindSscError::EmptyMovingImage {
                shape: moving_shape,
            });
        }
        let moving_geometry = CartesianGeometry::try_from_image(moving, "moving")?;
        let moving_values =
            moving
                .data_slice()
                .map_err(|error| MindSscError::NonContiguousImage {
                    image: "moving",
                    reason: error.to_string(),
                })?;
        let native = rigid_physical_affine_to_native::<B>(transform)?;
        validate_transform(&native)?;
        let mut loss = 0.0_f32;
        for (sample, (&center_index, &fixed_descriptor)) in
            self.indices.iter().zip(self.descriptors.iter()).enumerate()
        {
            let center = decode_index(center_index, self.shape)?;
            let moving_descriptor = descriptor_at(center, self.geometry, |support| {
                let fixed_world = self.fixed_geometry.index_to_world(support)?;
                let moving_world = apply_native_affine(&native, fixed_world);
                let moving_index = moving_geometry.world_to_index(moving_world)?;
                trilinear_background(moving_values, moving_shape, moving_index)
            })?;
            let weight = self.weights.as_deref().map_or(1.0, |values| values[sample]);
            let different_bits =
                f32::cast_from((fixed_descriptor ^ moving_descriptor).count_ones());
            loss = different_bits.mul_add(weight, loss);
        }
        // Both sums use the same positive weights, so the exact result lies in
        // [0, 1]. Close only the final f32 reduction roundoff at that contract.
        Ok((1.0 - loss / self.denominator).clamp(0.0, 1.0))
    }

    /// Fixed linear C-order indices used by every pose evaluation.
    #[must_use]
    pub fn selected_indices(&self) -> &[usize] {
        &self.indices
    }

    /// Exact heap-payload and algorithmic scratch accounting.
    pub fn memory_usage(&self) -> MindSscMemoryUsage {
        let index_bytes = size_of_val(&*self.indices);
        let descriptor_bytes = size_of_val(&*self.descriptors);
        let weight_bytes = self.weights.as_deref().map_or(0, size_of_val);
        MindSscMemoryUsage {
            selected_centers: self.indices.len(),
            index_bytes,
            descriptor_bytes,
            weight_bytes,
            heap_payload_bytes: index_bytes + descriptor_bytes + weight_bytes,
            per_center_scratch_bytes: size_of::<[f32; 6]>() + size_of::<[f32; 12]>(),
        }
    }
}

/// One-shot fixed-domain MIND-SSC similarity in `[0, 1]`.
///
/// Repeated optimizer evaluations should construct [`MindSscFixedPrep`] once.
///
/// # Errors
///
/// Returns [`MindSscError`] under the same conditions as
/// [`MindSscFixedPrep::try_new`] and [`MindSscFixedPrep::eval`].
pub fn mind_ssc_value<B>(
    fixed: &Image<f32, B, 3>,
    moving: &Image<f32, B, 3>,
    transform: &AffineTransform,
    config: MindSscConfig,
    mask: Option<&[bool]>,
    weights: Option<&[f32]>,
) -> Result<f32, MindSscError>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    MindSscFixedPrep::try_new(fixed, config, mask, weights)?.eval(moving, transform)
}

fn fixed_value(values: &[f32], shape: [usize; 3], index: [usize; 3]) -> Result<f32, MindSscError> {
    let linear = linear_index(index, shape)?;
    let value = *values.get(linear).ok_or(MindSscError::IndexOverflow)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MindSscError::NonFiniteImageSample {
            image: "fixed",
            index: linear,
            value,
        })
    }
}
