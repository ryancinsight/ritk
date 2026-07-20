//! Canonical Coeus-backed image contract.

use std::fmt;

use anyhow::{anyhow, bail};
use coeus_core::{ComputeBackend, CpuAddressableStorage, Scalar};
use coeus_tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};

/// Medical image backed by a Coeus tensor.
///
/// The `D` const generic is the image dimensionality. Construction validates
/// that the tensor rank matches `D`, so index-space metadata cannot be paired
/// with a tensor of a different rank.
#[derive(Clone)]
pub struct Image<T, B, const D: usize>
where
    T: Scalar,
    B: ComputeBackend,
{
    data: Tensor<T, B>,
    origin: Point<D>,
    spacing: Spacing<D>,
    direction: Direction<D>,
}

impl<T, B, const D: usize> fmt::Debug for Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image")
            .field("shape", &self.data.shape())
            .field("origin", &self.origin)
            .field("spacing", &self.spacing)
            .field("direction", &self.direction)
            .finish()
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
{
    /// Create an image from flat voxel data, shape, and physical metadata.
    ///
    /// This constructor validates the shape product before constructing the
    /// tensor, so malformed external buffers fail at the image boundary instead
    /// of relying on a downstream tensor panic.
    ///
    /// # Errors
    ///
    /// Returns an error when the checked product of `dims` overflows, or when
    /// `data.len()` does not equal that product.
    pub fn from_flat_on(
        data: Vec<T>,
        dims: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
        backend: &B,
    ) -> anyhow::Result<Self> {
        let expected = checked_numel(&dims)?;
        if data.len() != expected {
            bail!(
                "image flat data length {} does not match shape {:?} product {}",
                data.len(),
                dims,
                expected
            );
        }

        Self::new(
            Tensor::from_slice_on(dims, &data, backend),
            origin,
            spacing,
            direction,
        )
    }

    /// Create an image from Coeus tensor data and physical metadata.
    ///
    /// # Errors
    ///
    /// Returns an error when `data.ndim() != D`.
    pub fn new(
        data: Tensor<T, B>,
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> anyhow::Result<Self> {
        let rank = data.ndim();
        if rank != D {
            bail!("image tensor rank mismatch: expected {D}, got {rank}");
        }

        Ok(Self {
            data,
            origin,
            spacing,
            direction,
        })
    }

    /// Get the image data tensor.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &Tensor<T, B> {
        &self.data
    }

    /// Get the physical coordinate of the first pixel.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> &Point<D> {
        &self.origin
    }

    /// Get the physical distance between neighboring pixels.
    #[inline]
    #[must_use]
    pub fn spacing(&self) -> &Spacing<D> {
        &self.spacing
    }

    /// Get the direction cosine matrix for the image axes.
    #[inline]
    #[must_use]
    pub fn direction(&self) -> &Direction<D> {
        &self.direction
    }

    /// Get the image shape as a fixed-rank array.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> [usize; D] {
        self.data
            .shape()
            .try_into()
            .expect("invariant: Image::new validates tensor rank equals D")
    }

    /// Consume the image and return the underlying Coeus tensor.
    #[inline]
    #[must_use]
    pub fn into_tensor(self) -> Tensor<T, B> {
        self.data
    }

    /// Consume the image and return all components.
    ///
    /// Returns `(tensor, origin, spacing, direction)`.
    #[inline]
    #[must_use]
    pub fn into_parts(self) -> (Tensor<T, B>, Point<D>, Spacing<D>, Direction<D>) {
        (self.data, self.origin, self.spacing, self.direction)
    }

    /// Convert a physical-space point to a continuous image index.
    ///
    /// The mapping is `S^-1 D^-1 (point - origin)`, where `D` is the
    /// direction cosine matrix and `S` is the diagonal spacing matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when the direction matrix is singular.
    pub fn physical_point_to_continuous_index(&self, point: &Point<D>) -> anyhow::Result<Point<D>> {
        let inverse = self
            .direction
            .try_inverse()
            .ok_or_else(|| anyhow!("image direction matrix is singular"))?;
        let rotated = inverse * (*point - self.origin);
        let mut index = Point::origin();
        for axis in 0..D {
            index[axis] = rotated[axis] / self.spacing[axis];
        }
        Ok(index)
    }

    /// Convert a continuous image index to a physical-space point.
    ///
    /// The mapping is `origin + D S index`.
    #[must_use]
    pub fn continuous_index_to_physical_point(&self, index: &Point<D>) -> Point<D> {
        let mut scaled = ritk_spatial::Vector::zeros();
        for axis in 0..D {
            scaled[axis] = index[axis] * self.spacing[axis];
        }
        self.origin + self.direction * scaled
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: coeus_core::Float,
    B: coeus_ops::BackendOps<T> + Default,
{
    /// Map a `[point_count, D]` Coeus tensor from physical coordinates to
    /// continuous image indices.
    ///
    /// Coordinate columns use the same axis order as [`Point<D>`]. Backend
    /// dispatch occurs once in broadcast subtraction and matrix multiplication;
    /// the method does not materialize point data on the host.
    ///
    /// # Errors
    ///
    /// Returns an error when `points` is not rank two with trailing dimension
    /// `D`, or when the direction matrix is singular.
    pub fn physical_points_to_continuous_indices(
        &self,
        points: &Tensor<T, B>,
        backend: &B,
    ) -> anyhow::Result<Tensor<T, B>> {
        if points.ndim() != 2 || points.shape()[1] != D {
            bail!(
                "physical point tensor shape must be [point_count, {D}], got {:?}",
                points.shape()
            );
        }
        let inverse = self
            .direction
            .try_inverse()
            .ok_or_else(|| anyhow!("image direction matrix is singular"))?;
        let origin = (0..D)
            .map(|axis| T::from_f64(self.origin[axis]))
            .collect::<Vec<_>>();
        let matrix = (0..D)
            .flat_map(|input_axis| {
                (0..D).map(move |output_axis| {
                    T::from_f64(inverse[(output_axis, input_axis)] / self.spacing[output_axis])
                })
            })
            .collect::<Vec<_>>();
        let origin = Tensor::from_slice_on([1, D], &origin, backend);
        let matrix = Tensor::from_slice_on([D, D], &matrix, backend);
        let centered = coeus_ops::sub(points, &origin, backend);
        Ok(coeus_ops::matmul(&centered, &matrix, backend))
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend + Default,
{
    /// Create an image from flat voxel data on `B::default()`.
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as [`Image::from_flat_on`].
    #[inline]
    pub fn from_flat(
        data: Vec<T>,
        dims: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> anyhow::Result<Self> {
        Self::from_flat_on(data, dims, origin, spacing, direction, &B::default())
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    /// Borrow contiguous host-addressable image data.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is not row-major contiguous.
    #[inline]
    pub fn data_slice(&self) -> anyhow::Result<&[T]> {
        if !self.data.is_contiguous() {
            return Err(anyhow!(
                "image data is not contiguous: shape={:?}, strides={:?}",
                self.data.shape(),
                self.data.strides()
            ));
        }
        Ok(self.data.as_slice())
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
{
    /// Host image data in logical row-major order, borrowing when the tensor
    /// is already contiguous and materializing a compact copy otherwise.
    ///
    /// The layout-independent host-extraction surface format writers and
    /// boundary code need (ADR 0002 cutover prerequisite): unlike
    /// [`Self::data_slice`], it never fails on a strided view — it pays the
    /// copy exactly when the layout requires one (`Cow::Owned`), and is
    /// zero-copy otherwise (`Cow::Borrowed`). Mirrors the Coeus `Image`'s
    /// `data_slice() -> Cow` contract. (`B: Default` follows from
    /// `Tensor::to_contiguous_on`'s own bound.)
    #[must_use]
    pub fn data_cow_on(&self, backend: &B) -> std::borrow::Cow<'_, [T]> {
        self.data.host_cow_on(backend)
    }

    /// Owned host image data in logical row-major order (layout-independent).
    ///
    /// Thin wrapper over [`Self::data_cow_on`] for callers that need a `Vec`
    /// (the `Image` type's `try_data_vec` equivalent).
    #[must_use]
    pub fn data_vec_on(&self, backend: &B) -> Vec<T> {
        self.data_cow_on(backend).into_owned()
    }

    /// Copy logical row-major image data into an owned host buffer.
    ///
    /// # Errors
    ///
    /// This canonical Coeus image contract materializes backend storage and
    /// non-contiguous views, so extraction succeeds for every valid image.
    pub fn try_data_vec_on(&self, backend: &B) -> anyhow::Result<Vec<T>> {
        Ok(self.data.to_vec_on(backend))
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend + Default,
{
    /// [`Self::data_cow_on`] on `B::default()` (mirrors [`Self::from_flat`]).
    #[must_use]
    pub fn data_cow(&self) -> std::borrow::Cow<'_, [T]> {
        self.data_cow_on(&B::default())
    }

    /// [`Self::data_vec_on`] on `B::default()`.
    #[must_use]
    pub fn data_vec(&self) -> Vec<T> {
        self.data_vec_on(&B::default())
    }

    /// [`Self::try_data_vec_on`] on `B::default()`.
    pub fn try_data_vec(&self) -> anyhow::Result<Vec<T>> {
        self.try_data_vec_on(&B::default())
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
{
    /// Convert a continuous index to a physical point.
    ///
    /// `point = origin + Direction * (index * spacing)`
    ///
    /// Reproduces the exact arithmetic and column conventions of the
    /// `transform_continuous_index_to_physical_point` operation.
    pub fn transform_continuous_index_to_physical_point(
        &self,
        index: &ritk_spatial::Point<D>,
    ) -> ritk_spatial::Point<D> {
        let mut scaled_index = ritk_spatial::Vector::<D>::zeros();
        for i in 0..D {
            scaled_index[i] = index[i] * self.spacing()[i];
        }
        let rotated = *self.direction() * scaled_index;
        *self.origin() + rotated
    }

    /// Convert a continuous physical point to a continuous index.
    ///
    /// `index = (Direction^-1 * (point - origin)) / spacing`
    ///
    /// Reproduces the exact arithmetic and column conventions of the
    /// `transform_physical_point_to_continuous_index` operation.
    pub fn transform_physical_point_to_continuous_index(
        &self,
        point: &ritk_spatial::Point<D>,
    ) -> ritk_spatial::Point<D> {
        let diff = *point - *self.origin();
        let inv_dir = self
            .direction()
            .try_inverse()
            .expect("direction matrix must be invertible");
        let rotated = inv_dir * diff;
        let mut index = ritk_spatial::Point::<D>::origin();
        for i in 0..D {
            index[i] = rotated[i] / self.spacing()[i];
        }
        index
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    /// Batch transform physical points to continuous indices.
    ///
    /// Reproduces the `Image` type's `world_to_index_tensor`,
    /// reproducing its exact arithmetic and column conventions bit-faithfully.
    ///
    /// # Conventions
    ///
    /// - Input `points` is `[N, D]` with columns in **axis-major** order
    ///   (column `a` = spatial axis `a`, the same order as `origin`/`spacing`).
    /// - Output indices are `[N, D]` with columns in **innermost-first** order
    ///   (column `c` = spatial axis `D-1-c`, i.e. column 0 = x = axis `D-1`),
    ///   matching `grid::generate_grid` and the interpolation kernels.
    /// - Per point: `index_axis = (Direction^-1 · (point − origin)) ⊘ spacing`,
    ///   emitted innermost-first. Arithmetic runs in `T` (the metadata-derived
    ///   `Direction^-1 / spacing` and `origin` are narrowed from `f64` to `T`
    ///   once, matching the Coeus path's `as f32` cast before the batched apply).
    ///
    /// # Panics
    ///
    /// Panics when `points` is not rank-2 or its trailing dimension is not `D`
    /// (a batch-shape precondition; callers pass `[N, D]` point grids).
    #[must_use]
    pub fn world_to_index_native_on(&self, points: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        let n = self.assert_batch_shape(points);

        let inv_dir = self
            .direction()
            .try_inverse()
            .expect("invariant: direction matrix must be invertible");

        // t[r][c] maps axis-major input column r to innermost-first output column
        // c (axis = D-1-c): t[r][c] = inv_dir[(axis, r)] / spacing[axis]. The
        // division is performed in f64 then narrowed to T, matching the Coeus
        // matrix build's `as f32`.
        let mut t = [[T::zero(); D]; D];
        for (r, row) in t.iter_mut().enumerate() {
            for (c, cell) in row.iter_mut().enumerate() {
                let axis = D - 1 - c;
                *cell = T::from_f64(inv_dir[(axis, r)] / self.spacing()[axis]);
            }
        }
        let origin_t = self.origin_narrowed();

        let src = points.as_slice();
        let mut out = vec![T::zero(); n * D];
        for (p, o) in src.chunks_exact(D).zip(out.chunks_exact_mut(D)) {
            for (c, oc) in o.iter_mut().enumerate() {
                let mut acc = T::zero();
                for r in 0..D {
                    acc += (p[r] - origin_t[r]) * t[r][c];
                }
                *oc = acc;
            }
        }

        Tensor::from_slice_on([n, D], &out, backend)
    }

    /// Batch transform continuous indices to physical points.
    ///
    /// Reproduces the `Image` type's `index_to_world_tensor`,
    /// reproducing its exact arithmetic and column conventions bit-faithfully.
    ///
    /// # Conventions
    ///
    /// - Input `indices` is `[N, D]` with columns in **innermost-first** order
    ///   (column `r` = spatial axis `D-1-r`), matching `grid::generate_grid`.
    /// - Output points are `[N, D]` with columns in **axis-major** order
    ///   (column `a` = spatial axis `a`, the same order as `origin`).
    /// - Per point: `point = origin + Direction · (index ⊙ spacing)`, consuming
    ///   the innermost-first index columns. Arithmetic runs in `T` (the
    ///   metadata-derived `spacing · Direction` and `origin` are narrowed from
    ///   `f64` to `T` once, matching the Coeus path's `as f32` cast).
    ///
    /// # Panics
    ///
    /// Panics when `indices` is not rank-2 or its trailing dimension is not `D`.
    #[must_use]
    pub fn index_to_world_native_on(&self, indices: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        let n = self.assert_batch_shape(indices);

        // m[r][c] maps innermost-first index column r (axis = D-1-r) to axis-major
        // output column c: m[r][c] = spacing[axis] * direction[(c, axis)]. Product
        // in f64 then narrowed to T, matching the Coeus matrix build's `as f32`.
        let mut m = [[T::zero(); D]; D];
        for (r, row) in m.iter_mut().enumerate() {
            let axis = D - 1 - r;
            for (c, cell) in row.iter_mut().enumerate() {
                *cell = T::from_f64(self.spacing()[axis] * self.direction()[(c, axis)]);
            }
        }
        let origin_t = self.origin_narrowed();

        let src = indices.as_slice();
        let mut out = vec![T::zero(); n * D];
        for (idx, o) in src.chunks_exact(D).zip(out.chunks_exact_mut(D)) {
            for (c, oc) in o.iter_mut().enumerate() {
                let mut acc = T::zero();
                for r in 0..D {
                    acc += idx[r] * m[r][c];
                }
                *oc = acc + origin_t[c];
            }
        }

        Tensor::from_slice_on([n, D], &out, backend)
    }

    /// Narrow the `f64` origin into `T` once (mirrors the Coeus path's `as f32`).
    fn origin_narrowed(&self) -> [T; D] {
        let mut origin_t = [T::zero(); D];
        for (i, o) in origin_t.iter_mut().enumerate() {
            *o = T::from_f64(self.origin()[i]);
        }
        origin_t
    }

    /// Validate the `[N, D]` batch shape and return `N`.
    fn assert_batch_shape(&self, points: &Tensor<T, B>) -> usize {
        let shape = points.shape();
        assert_eq!(
            shape.len(),
            2,
            "batch point transform requires a rank-2 [N, D] tensor, got rank {}",
            shape.len()
        );
        assert_eq!(
            shape[1], D,
            "batch point transform trailing dimension {} does not match image dimensionality {D}",
            shape[1]
        );
        shape[0]
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    /// [`Self::world_to_index_native_on`] on `B::default()`.
    ///
    /// The single-argument form that most directly replaces the Coeus
    /// `world_to_index_tensor` at call sites.
    #[inline]
    #[must_use]
    pub fn world_to_index_native(&self, points: &Tensor<T, B>) -> Tensor<T, B> {
        self.world_to_index_native_on(points, &B::default())
    }

    /// [`Self::index_to_world_native_on`] on `B::default()`.
    ///
    /// The single-argument form that most directly replaces the Coeus
    /// `index_to_world_tensor` at call sites.
    #[inline]
    #[must_use]
    pub fn index_to_world_native(&self, indices: &Tensor<T, B>) -> Tensor<T, B> {
        self.index_to_world_native_on(indices, &B::default())
    }
}

fn checked_numel(dims: &[usize]) -> anyhow::Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| anyhow!("image shape {:?} product overflows usize", dims))
    })
}

#[cfg(test)]
mod tests {
    use coeus_core::SequentialBackend;

    use super::*;

    type TensorImage<const D: usize> = Image<f32, SequentialBackend, D>;

    fn metadata_2d() -> (Point<2>, Spacing<2>, Direction<2>) {
        (
            Point::new([10.0, 20.0]),
            Spacing::new([0.5, 1.5]),
            Direction::identity(),
        )
    }

    #[test]
    fn physical_index_mapping_obeys_anisotropic_rotated_reference() {
        let image = TensorImage::<3>::from_flat(
            vec![0.0; 6],
            [2, 3, 1],
            Point::new([10.0, 20.0, 0.0]),
            Spacing::new([2.0, 4.0, 1.0]),
            Direction::from_row_major([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        )
        .expect("fixture shape and data length agree");
        let index = Point::new([3.0, -2.0, 0.0]);
        let physical = image.continuous_index_to_physical_point(&index);
        assert_eq!(physical, Point::new([18.0, 26.0, 0.0]));
        assert_eq!(
            image
                .physical_point_to_continuous_index(&physical)
                .expect("rotation matrix is invertible"),
            index
        );
    }

    #[test]
    fn physical_index_mapping_rejects_singular_direction() {
        let image = TensorImage::<3>::from_flat(
            vec![0.0],
            [1, 1, 1],
            Point::origin(),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::from_row_major([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        )
        .expect("fixture shape and data length agree");
        let error = image
            .physical_point_to_continuous_index(&Point::origin())
            .unwrap_err();
        assert_eq!(error.to_string(), "image direction matrix is singular");
    }

    #[test]
    fn physical_point_tensor_matches_scalar_mapping() {
        let image = TensorImage::<3>::from_flat(
            vec![0.0; 8],
            [2, 2, 2],
            Point::new([10.0, 20.0, 30.0]),
            Spacing::new([2.0, 4.0, 5.0]),
            Direction::identity(),
        )
        .expect("fixture shape and data length agree");
        let backend = SequentialBackend;
        let points =
            Tensor::from_slice_on([2, 3], &[12.0_f32, 28.0, 40.0, 8.0, 16.0, 25.0], &backend);
        let indices = image
            .physical_points_to_continuous_indices(&points, &backend)
            .expect("identity direction and point tensor are valid");
        assert_eq!(indices.as_slice(), &[1.0, 2.0, 2.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn physical_point_tensor_rejects_wrong_coordinate_width() {
        let image = TensorImage::<3>::from_flat(
            vec![0.0],
            [1, 1, 1],
            Point::origin(),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
        .expect("fixture shape and data length agree");
        let backend = SequentialBackend;
        let points = Tensor::zeros_on([2, 2], &backend);
        let error = match image.physical_points_to_continuous_indices(&points, &backend) {
            Ok(_) => panic!("wrong coordinate width must be rejected"),
            Err(error) => error,
        };
        assert_eq!(
            error.to_string(),
            "physical point tensor shape must be [point_count, 3], got [2, 2]"
        );
    }

    #[test]
    fn construction_preserves_shape_and_metadata() {
        let data =
            Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (origin, spacing, direction) = metadata_2d();

        let image = TensorImage::<2>::new(data, origin, spacing, direction).unwrap();

        assert_eq!(image.shape(), [2, 3]);
        assert_eq!(image.origin(), &origin);
        assert_eq!(image.spacing(), &spacing);
        assert_eq!(image.direction(), &direction);
        assert_eq!(image.data_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn from_flat_preserves_shape_values_and_metadata() {
        let (origin, spacing, direction) = metadata_2d();

        let image = TensorImage::<2>::from_flat(
            vec![1.0, 2.0, 3.0, 4.0],
            [2, 2],
            origin,
            spacing,
            direction,
        )
        .unwrap();

        assert_eq!(image.shape(), [2, 2]);
        assert_eq!(image.origin(), &origin);
        assert_eq!(image.spacing(), &spacing);
        assert_eq!(image.direction(), &direction);
        assert_eq!(image.data_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn data_cow_borrows_when_contiguous() {
        let (origin, spacing, direction) = metadata_2d();
        let image = TensorImage::<2>::from_flat(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            origin,
            spacing,
            direction,
        )
        .unwrap();

        let cow = image.data_cow();
        assert!(
            matches!(cow, std::borrow::Cow::Borrowed(_)),
            "contiguous image must borrow (zero-copy)"
        );
        assert_eq!(cow.as_ref(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(image.data_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn data_cow_materializes_logical_order_for_permuted_view() {
        // Build a [2, 3] image, permute the tensor to [3, 2] (non-contiguous
        // strided view), and re-wrap. Logical row-major order of the permuted
        // view is the host transpose — the oracle the extraction must match.
        let (origin, spacing, direction) = metadata_2d();
        let base =
            Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let permuted = base.permute(&[1, 0]); // shape [3, 2], strides non-contiguous
        let image = TensorImage::<2>::new(permuted, origin, spacing, direction).unwrap();

        // The strict borrow API must refuse the strided view (existing contract).
        assert!(
            image.data_slice().is_err(),
            "data_slice must reject non-contiguous"
        );

        // Host transpose oracle: [[1,4],[2,5],[3,6]] row-major.
        let expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let cow = image.data_cow();
        assert!(
            matches!(cow, std::borrow::Cow::Owned(_)),
            "non-contiguous image must materialize (owned)"
        );
        assert_eq!(cow.as_ref(), &expected);
        assert_eq!(image.data_vec(), expected.to_vec());
    }

    #[test]
    fn from_flat_rejects_shape_product_mismatch() {
        let (origin, spacing, direction) = metadata_2d();

        let err =
            TensorImage::<2>::from_flat(vec![1.0, 2.0, 3.0], [2, 2], origin, spacing, direction)
                .unwrap_err();

        assert_eq!(
            err.to_string(),
            "image flat data length 3 does not match shape [2, 2] product 4"
        );
    }

    #[test]
    fn from_flat_rejects_shape_product_overflow() {
        let (origin, spacing, direction) = metadata_2d();

        let err =
            TensorImage::<2>::from_flat(Vec::new(), [usize::MAX, 2], origin, spacing, direction)
                .unwrap_err();

        assert_eq!(
            err.to_string(),
            "image shape [18446744073709551615, 2] product overflows usize"
        );
    }

    #[test]
    fn construction_rejects_rank_mismatch() {
        let data = Tensor::<f32, SequentialBackend>::from_slice([2, 3, 1], &[0.0; 6]);
        let (origin, spacing, direction) = metadata_2d();

        let err = TensorImage::<2>::new(data, origin, spacing, direction).unwrap_err();

        assert_eq!(
            err.to_string(),
            "image tensor rank mismatch: expected 2, got 3"
        );
    }

    #[test]
    fn data_slice_rejects_non_contiguous_layout() {
        let data =
            Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let column_view = Tensor::from_raw_parts(
            data.storage().clone(),
            data.layout().slice(&[(0, 2), (1, 2)]),
        );
        let (origin, spacing, direction) = metadata_2d();
        let image = TensorImage::<2>::new(column_view, origin, spacing, direction).unwrap();

        let err = image.data_slice().unwrap_err();

        assert_eq!(
            err.to_string(),
            "image data is not contiguous: shape=[2, 1], strides=[3, 1]"
        );
    }

    // ── Batch point transforms ──────────────────────────────────────────────

    /// Rotation of 90° about the z-axis: rows [0,-1,0], [1,0,0], [0,0,1].
    /// Orthonormal (inverse = transpose), determinant 1 — exercises the
    /// direction terms and the axis-major ↔ innermost-first column reordering.
    fn rotated_metadata_3d() -> (Point<3>, Spacing<3>, Direction<3>) {
        (
            Point::new([10.5, -3.25, 7.0]),
            Spacing::new([0.5, 1.25, 2.0]),
            Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        )
    }

    /// Deterministic pseudo-random point rows in `[-range, range]` (LCG; no
    /// test-time randomness dependency, replayable).
    fn pseudo_points(n: usize, range: f64) -> Vec<f64> {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        (0..n * 3)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 33) as f64) / ((1u64 << 31) as f64); // [0, 2)
                (unit - 1.0) * range
            })
            .collect()
    }

    fn dummy_image<T: Scalar, const D: usize>(
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> Image<T, SequentialBackend, D> {
        let dims = [1usize; D];
        Image::from_flat(vec![T::zero()], dims, origin, spacing, direction).unwrap()
    }

    /// Analytical oracle — identity geometry. `world_to_index` maps a physical
    /// point (axis-major columns) to its index (innermost-first columns), which
    /// under identity origin/spacing/direction equals the point with reversed
    /// column order; `index_to_world` is the exact inverse. Exact in f64 (×1,
    /// +0 only).
    #[test]
    fn native_batch_identity_reverses_columns_exactly() {
        let img = dummy_image::<f64, 3>(
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );

        let world = [1.0_f64, 2.0, 3.0, -4.0, 5.5, -6.25]; // 2 axis-major points
        let world_t = Tensor::<f64, SequentialBackend>::from_slice([2, 3], &world);

        let idx = img.world_to_index_native(&world_t);
        // innermost-first == axis-major reversed per row.
        assert_eq!(idx.as_slice(), &[3.0, 2.0, 1.0, -6.25, 5.5, -4.0]);

        let back = img.index_to_world_native(&idx);
        assert_eq!(back.as_slice(), &world);
    }

    /// Independent oracle — the batch transforms agree with the single-point
    /// `transform_*` methods (mathematically independent code path) under
    /// non-trivial anisotropic, rotated geometry. Accounts for the batch
    /// innermost-first index column order vs the single-point axis-major
    /// `Point`. f64 throughout; tolerance is f64 machine slack.
    #[test]
    fn native_batch_agrees_with_single_point_methods() {
        let (origin, spacing, direction) = rotated_metadata_3d();
        let img = dummy_image::<f64, 3>(origin, spacing, direction);

        let pts = pseudo_points(12, 40.0);
        let world_t = Tensor::<f64, SequentialBackend>::from_slice([12, 3], &pts);
        let idx_batch = img.world_to_index_native(&world_t);

        for row in 0..12 {
            let p = Point::<3>::new([pts[row * 3], pts[row * 3 + 1], pts[row * 3 + 2]]);
            // Single-point index (axis-major).
            let idx_axis = img.transform_physical_point_to_continuous_index(&p);
            let batch_row = &idx_batch.as_slice()[row * 3..row * 3 + 3];
            // batch column c ↔ axis D-1-c.
            for c in 0..3 {
                assert!(
                    (batch_row[c] - idx_axis[2 - c]).abs() <= 1e-9,
                    "row {row} col {c}: batch={}, single={}",
                    batch_row[c],
                    idx_axis[2 - c]
                );
            }

            // index → world: feed the batch (innermost-first) index back.
            let world_batch = img.index_to_world_native(
                &Tensor::<f64, SequentialBackend>::from_slice([1, 3], batch_row),
            );
            let idx_pt = Point::<3>::new([idx_axis[0], idx_axis[1], idx_axis[2]]);
            let world_single = img.transform_continuous_index_to_physical_point(&idx_pt);
            for a in 0..3 {
                assert!(
                    (world_batch.as_slice()[a] - world_single[a]).abs() <= 1e-9,
                    "row {row} axis {a}: batch={}, single={}",
                    world_batch.as_slice()[a],
                    world_single[a]
                );
            }
        }
    }

    /// Round-trip: index → world → index recovers the original index within f64
    /// eps under non-trivial geometry (composition consistency of the pair).
    #[test]
    fn native_batch_index_world_roundtrip_identity() {
        let (origin, spacing, direction) = rotated_metadata_3d();
        let img = dummy_image::<f64, 3>(origin, spacing, direction);

        let idx = pseudo_points(20, 30.0); // innermost-first index rows
        let idx_t = Tensor::<f64, SequentialBackend>::from_slice([20, 3], &idx);

        let world = img.index_to_world_native(&idx_t);
        let world_t = Tensor::<f64, SequentialBackend>::from_slice([20, 3], world.as_slice());
        let idx_rt = img.world_to_index_native(&world_t);

        for (a, b) in idx.iter().zip(idx_rt.as_slice()) {
            assert!((a - b).abs() <= 1e-9, "round-trip drift: {a} vs {b}");
        }
    }

    #[test]
    fn into_parts_returns_exact_tensor_and_metadata() {
        let data = Tensor::<f32, SequentialBackend>::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let (origin, spacing, direction) = metadata_2d();
        let image = TensorImage::<2>::new(data, origin, spacing, direction).unwrap();

        let (data, returned_origin, returned_spacing, returned_direction) = image.into_parts();

        assert_eq!(data.shape(), &[2, 2]);
        assert_eq!(data.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(returned_origin, origin);
        assert_eq!(returned_spacing, spacing);
        assert_eq!(returned_direction, direction);
    }
}
