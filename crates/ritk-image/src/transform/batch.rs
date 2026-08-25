//! Batch coordinate transforms over `[point_count, D]` tensors.
//!
//! Backend dispatch happens once per call rather than once per point, so the
//! per-[`CoordinateMap`] kernels here operate on whole tensors and never
//! materialise coordinates on the host. Columns are innermost-first, unlike
//! the axis-indexed form in [`super::point`].

use anyhow::{anyhow, bail};
use coeus_core::{ComputeBackend, CpuAddressableStorage, Scalar};
use coeus_tensor::Tensor;
use ritk_spatial::CoordinateMap;

use crate::types::Image;

impl<T, B, const D: usize> Image<T, B, D>
where
    T: coeus_core::Float,
    B: coeus_ops::BackendOps<T> + Default,
{
    /// Map a `[point_count, D]` Coeus tensor from physical coordinates to
    /// continuous image indices.
    ///
    /// Coordinate columns use the same axis order as [`ritk_spatial::Point`]. Backend
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
    ///
    /// Dispatch on [`CoordinateMap`] happens **once** here; each arm then runs
    /// its own monomorphic per-point loop.
    #[must_use]
    pub fn world_to_index_native_on(&self, points: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        let n = self.assert_batch_shape(points);
        match &self.map {
            CoordinateMap::Cartesian => self.world_to_index_cartesian(points, n, backend),
            CoordinateMap::CurvilinearArray(geometry) => {
                self.world_to_index_curvilinear(geometry, points, n, backend)
            }
            CoordinateMap::PhasedArray3D(geometry) => {
                self.world_to_index_phased_array(geometry, points, n, backend)
            }
            CoordinateMap::SliceSeries(sweep) => {
                self.world_to_index_slice_series(sweep, points, n, backend)
            }
        }
    }

    /// Cartesian arm of [`Self::world_to_index_native_on`].
    #[must_use]
    fn world_to_index_cartesian(
        &self,
        points: &Tensor<T, B>,
        n: usize,
        backend: &B,
    ) -> Tensor<T, B> {
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
    ///
    /// Dispatch on [`CoordinateMap`] happens **once** here; each arm then runs
    /// its own monomorphic per-point loop.
    #[must_use]
    pub fn index_to_world_native_on(&self, indices: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        let n = self.assert_batch_shape(indices);
        match &self.map {
            CoordinateMap::Cartesian => self.index_to_world_cartesian(indices, n, backend),
            CoordinateMap::CurvilinearArray(geometry) => {
                self.index_to_world_curvilinear(geometry, indices, n, backend)
            }
            CoordinateMap::PhasedArray3D(geometry) => {
                self.index_to_world_phased_array(geometry, indices, n, backend)
            }
            CoordinateMap::SliceSeries(sweep) => {
                self.index_to_world_slice_series(sweep, indices, n, backend)
            }
        }
    }

    /// Cartesian arm of [`Self::index_to_world_native_on`].
    #[must_use]
    fn index_to_world_cartesian(
        &self,
        indices: &Tensor<T, B>,
        n: usize,
        backend: &B,
    ) -> Tensor<T, B> {
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

    /// Curvilinear arm of [`Self::index_to_world_native_on`].
    ///
    /// The two innermost spatial axes carry the polar pair: index column 0 is
    /// the sample along a beam and column 1 the beam number, mapping to
    /// `axis D-1 = r·sin θ` and `axis D-2 = r·cos θ`. Any outer axes use the
    /// same affine row as the Cartesian arm.
    ///
    /// Trigonometry runs in `f64` and narrows to `T` per point, because the
    /// polar map is not expressible as a hoisted matrix the way the affine one
    /// is.
    #[must_use]
    fn index_to_world_curvilinear(
        &self,
        geometry: &ritk_spatial::CurvilinearArray,
        indices: &Tensor<T, B>,
        n: usize,
        backend: &B,
    ) -> Tensor<T, B> {
        // Affine rows for the outer axes, hoisted exactly as the Cartesian arm.
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
            let (radius, angle) =
                geometry.polar_from_index(Scalar::to_f64(idx[0]), Scalar::to_f64(idx[1]));
            o[D - 1] = T::from_f64(radius * angle.sin());
            o[D - 2] = T::from_f64(radius * angle.cos());
            for c in 0..D - 2 {
                let mut acc = T::zero();
                for r in 0..D {
                    acc += idx[r] * m[r][c];
                }
                o[c] = acc + origin_t[c];
            }
        }

        Tensor::from_slice_on([n, D], &out, backend)
    }

    /// Curvilinear arm of [`Self::world_to_index_native_on`].
    ///
    /// Inverse of [`Self::index_to_world_curvilinear`]. Points outside the
    /// acquisition half-plane have no beam and are emitted as NaN in the two
    /// polar columns, so an out-of-fan sample is unmistakable at the caller
    /// rather than silently aliasing onto a real beam. Callers that resample
    /// treat a non-finite index as background, exactly as they already do for
    /// an index outside the image bounds.
    #[must_use]
    fn world_to_index_curvilinear(
        &self,
        geometry: &ritk_spatial::CurvilinearArray,
        points: &Tensor<T, B>,
        n: usize,
        backend: &B,
    ) -> Tensor<T, B> {
        let inv_dir = self
            .direction()
            .try_inverse()
            .expect("invariant: direction matrix must be invertible");
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
            let lateral = Scalar::to_f64(p[D - 1]);
            let axial = Scalar::to_f64(p[D - 2]);
            match geometry.index_from_cartesian(lateral, axial) {
                Some((sample, beam)) => {
                    o[0] = T::from_f64(sample);
                    o[1] = T::from_f64(beam);
                }
                None => {
                    o[0] = T::from_f64(f64::NAN);
                    o[1] = T::from_f64(f64::NAN);
                }
            }
            for c in 2..D {
                let mut acc = T::zero();
                for r in 0..D {
                    acc += (p[r] - origin_t[r]) * t[r][c];
                }
                o[c] = acc;
            }
        }

        Tensor::from_slice_on([n, D], &out, backend)
    }

    /// Phased-array arm of [`Self::index_to_world_native_on`].
    ///
    /// Beam-space index → probe-frame Cartesian → world. The probe frame is
    /// placed in world space by `origin + Direction · probe_point`, composing
    /// the phased-array steering geometry with the image's affine outer
    /// transform (atlas US-023-A2 P1 closure).
    ///
    /// Index columns are `(azimuth beam, elevation beam, sample)`; the physical
    /// triple lands on axes `(2, 1, 0)` — azimuth, elevation, depth — matching
    /// the innermost-first column convention. Only defined at `D == 3`, which
    /// [`CoordinateMap::validate_dimensionality`] enforces at attach time.
    ///
    /// A ray whose steering angle has no finite depth yields NaN, on the same
    /// grounds as the curvilinear out-of-fan case.
    #[must_use]
    fn index_to_world_phased_array(
        &self,
        geometry: &ritk_spatial::PhasedArray3D,
        indices: &Tensor<T, B>,
        n: usize,
        backend: &B,
    ) -> Tensor<T, B> {
        let dir = self.direction();
        let origin_t = self.origin_narrowed();
        let src = indices.as_slice();
        let mut out = vec![T::zero(); n * D];
        for (idx, o) in src.chunks_exact(D).zip(out.chunks_exact_mut(D)) {
            match geometry.cartesian_from_index(
                Scalar::to_f64(idx[0]),
                Scalar::to_f64(idx[1]),
                Scalar::to_f64(idx[2]),
            ) {
                Some((azimuth_axis, elevation_axis, depth)) => {
                    // probe_point in image axis order: axis 2=azimuth, 1=elevation, 0=depth
                    let probe = [depth, elevation_axis, azimuth_axis];
                    // world = origin + Direction · probe_point (axis-major)
                    for c in 0..D {
                        let mut acc = origin_t[c];
                        for r in 0..D {
                            acc += T::from_f64(dir[(c, r)]) * T::from_f64(probe[r]);
                        }
                        o[c] = acc;
                    }
                }
                None => {
                    for value in o.iter_mut() {
                        *value = T::from_f64(f64::NAN);
                    }
                }
            }
        }

        Tensor::from_slice_on([n, D], &out, backend)
    }

    /// Phased-array arm of [`Self::world_to_index_native_on`].
    ///
    /// World → probe frame via `Direction^-1 · (world - origin)` → beam-space
    /// index via phased-array geometry (atlas US-023-A2 P1 closure).
    ///
    /// Inverse of [`Self::index_to_world_phased_array`]. Points behind the
    /// array (`depth <= 0`) have no ray and are emitted as NaN.
    #[must_use]
    fn world_to_index_phased_array(
        &self,
        geometry: &ritk_spatial::PhasedArray3D,
        points: &Tensor<T, B>,
        n: usize,
        backend: &B,
    ) -> Tensor<T, B> {
        let inv_dir = self
            .direction()
            .try_inverse()
            .expect("invariant: direction matrix must be invertible");
        let origin_t = self.origin_narrowed();
        let src = points.as_slice();
        let mut out = vec![T::zero(); n * D];
        for (p, o) in src.chunks_exact(D).zip(out.chunks_exact_mut(D)) {
            // probe_point = Direction^-1 · (world - origin)
            let mut probe = [0.0f64; 3];
            for r in 0..D {
                let mut acc = 0.0f64;
                for c in 0..D {
                    acc += inv_dir[(r, c)] * (Scalar::to_f64(p[c]) - Scalar::to_f64(origin_t[c]));
                }
                probe[r] = acc;
            }
            // probe is in axis order [depth, elevation, azimuth]
            match geometry.index_from_cartesian(probe[2], probe[1], probe[0]) {
                Some((azimuth_index, elevation_index, sample)) => {
                    o[0] = T::from_f64(azimuth_index);
                    o[1] = T::from_f64(elevation_index);
                    o[2] = T::from_f64(sample);
                }
                None => {
                    for value in o.iter_mut() {
                        *value = T::from_f64(f64::NAN);
                    }
                }
            }
        }

        Tensor::from_slice_on([n, D], &out, backend)
    }

    /// Slice-series arm of [`Self::index_to_world_native_on`].
    ///
    /// Column 0 (innermost, `axis D-1`) = in-plane x, column 1 = in-plane y,
    /// column 2 (outermost, `axis D-3`) = slice index. Out-of-range slice
    /// indices are clamped per the forward-clamp convention.
    #[must_use]
    fn index_to_world_slice_series(
        &self,
        sweep: &ritk_spatial::SliceSeries,
        indices: &Tensor<T, B>,
        n: usize,
        backend: &B,
    ) -> Tensor<T, B> {
        let src = indices.as_slice();
        let mut out = vec![T::zero(); n * D];
        for (idx, o) in src.chunks_exact(D).zip(out.chunks_exact_mut(D)) {
            let j_x = Scalar::to_f64(idx[0]);
            let j_y = Scalar::to_f64(idx[1]);
            let slice_f = Scalar::to_f64(idx[2]);
            let world = sweep.world_from_index(j_x, j_y, slice_f);
            // Write axis-major: column c = spatial axis c.
            o[D - 1] = T::from_f64(world[0]);
            o[D - 2] = T::from_f64(world[1]);
            o[D - 3] = T::from_f64(world[2]);
        }
        Tensor::from_slice_on([n, D], &out, backend)
    }

    /// Slice-series arm of [`Self::world_to_index_native_on`].
    ///
    /// Points outside the sweep extent are emitted as NaN (rejection
    /// convention, matching the other non-Cartesian variants).
    #[must_use]
    fn world_to_index_slice_series(
        &self,
        sweep: &ritk_spatial::SliceSeries,
        points: &Tensor<T, B>,
        n: usize,
        backend: &B,
    ) -> Tensor<T, B> {
        let src = points.as_slice();
        let mut out = vec![T::zero(); n * D];
        for (p, o) in src.chunks_exact(D).zip(out.chunks_exact_mut(D)) {
            // Input is axis-major: column a = spatial axis a.
            let world = [
                Scalar::to_f64(p[D - 1]),
                Scalar::to_f64(p[D - 2]),
                Scalar::to_f64(p[D - 3]),
            ];
            match sweep.index_from_world(world) {
                Some(idx) => {
                    o[0] = T::from_f64(idx[0]);
                    o[1] = T::from_f64(idx[1]);
                    o[2] = T::from_f64(idx[2]);
                }
                None => {
                    for value in o.iter_mut() {
                        *value = T::from_f64(f64::NAN);
                    }
                }
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
