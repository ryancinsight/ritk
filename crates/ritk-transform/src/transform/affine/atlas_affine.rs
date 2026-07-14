//! Atlas-keyed sister module for affine transforms.
//!
//! Per `docs/adr/0012-ritk-burn-trait-rebind.md` §Decision §Sub-batch #3.f:
//! atlas-side parallel module exposing [`AtlasAffineTransform<B, D>`] sister
//! struct that mirrors the legacy `super::AffineTransform<B, D>`
//! (Burn-keyed surface, `coeus_nn::Module`-derive in legacy impl).
//!
//! The Atlas sister is a **structural twin**: it carries the same matrix /
//! translation / center fields but stores them as plain `Vec<f32>` host
//! slices (no `Param<Tensor>` / `Module` derive) so the compute step can
//! route through the canonical host-slice path. `coeus_nn::Module::forward`
//! deep contracts are reserved for sub-batch #5 \[major\] (the actual
//! neural-network forward pass migration); sub-batch #3.f exposes the
//! data-shape surface only.
//!
//! Strictly additive on production surface per the sub-batch #3.f
//! atomic-boundary invariant: every public symbol of `super::affine` is
//! preserved verbatim, alongside the sibling `rigid` / `scale` /
//! `translation` / `versor` modules.
//!
//! **No `Cargo.toml` mutation** beyond additive `coeus-core` +
//! `coeus-tensor` lines (the crate previously had neither). To avoid a
//! `thiserror` dep-add, the [`AtlasAffineError`] enum derives
//! `Debug+Clone+PartialEq+Eq` only and carries a hand-rolled `Display`
//! impl via [`std::fmt::Display`] + [`std::error::Error`] — zero new
//! crate-graph edges in this file.

use coeus_core::{ComputeBackend, CpuAddressableStorage, MoiraiBackend};
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::native as tensor_ops;

// ── Sister struct ─────────────────────────────────────────────────────────

/// Atlas-side sister struct to `AffineTransform`.
///
/// Field shape mirrors the legacy (matrix / translation / center) but
/// stored on plain host slices (`Vec<f32>`) so the construct-time path
/// stays compute-backend-agnostic. **No** `coeus_nn::Module` derive —
/// deep forward contracts are reserved for sub-batch #5 \[major\] per ADR
/// 0012 §Decision §3.
#[derive(Debug, Clone)]
pub struct AtlasAffineTransform<B: ComputeBackend, const D: usize> {
    /// `[D, D]` linear transformation matrix stored row-major on a host slice.
    matrix: Vec<f32>,
    /// `[D]` translation vector.
    translation: Vec<f32>,
    /// `[D]` fixed center of rotation / scaling.
    center: Vec<f32>,
    /// Phantom marker for the Atlas-typed `ComputeBackend`.
    _backend: std::marker::PhantomData<B>,
}

// ── Error semantics ───────────────────────────────────────────────────────

/// Atlas-side error variants for affine math. Carries the actual lengths
/// to make the error text containable by callers (the Display impl emits
/// the expected + actual lengths in `[N]` form).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtlasAffineError {
    /// Construct-time matrix length mismatch.
    MatrixShape {
        actual_len: usize,
        expected_len: usize,
    },
    /// Construct-time translation length mismatch.
    TranslationLength {
        actual_len: usize,
        expected_len: usize,
    },
    /// Construct-time center length mismatch.
    CenterLength {
        actual_len: usize,
        expected_len: usize,
    },
    /// Host-slice extract failed (non-CPU-resident device buffer).
    Extract(String),
    /// Atlas-side carrier construction failed.
    Construct(String),
}

impl std::fmt::Display for AtlasAffineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MatrixShape { actual_len, expected_len } => write!(
                f,
                "AtlasAffineTransform::new matrix length mismatch: expected [{expected_len}] elements ([D, D] for D), got [{actual_len}] elements"
            ),
            Self::TranslationLength { actual_len, expected_len } => write!(
                f,
                "AtlasAffineTransform::new translation length mismatch: expected [{expected_len}] elements ([D] for D), got [{actual_len}] elements"
            ),
            Self::CenterLength { actual_len, expected_len } => write!(
                f,
                "AtlasAffineTransform::new center length mismatch: expected [{expected_len}] elements ([D] for D), got [{actual_len}] elements"
            ),
            Self::Extract(s) => write!(f, "atlas affine transform_points extract failed: {s}"),
            Self::Construct(s) => write!(f, "atlas affine transform_points carrier construction failed: {s}"),
        }
    }
}

impl std::error::Error for AtlasAffineError {}

// ── Constructors ──────────────────────────────────────────────────────────

impl<B: ComputeBackend, const D: usize> AtlasAffineTransform<B, D> {
    /// Construct a new Atlas affine transform from host-slice params.
    ///
    /// Sister to `AffineTransform::new`. The matrix is stored
    /// row-major as a flat `&[f32]` slice of length `D * D`; translation
    /// and center are `&[f32]` slices of length `D`.
    pub fn try_new(
        matrix: &[f32],
        translation: &[f32],
        center: &[f32],
    ) -> Result<Self, AtlasAffineError> {
        let expected = D * D;
        if matrix.len() != expected {
            return Err(AtlasAffineError::MatrixShape {
                actual_len: matrix.len(),
                expected_len: expected,
            });
        }
        if translation.len() != D {
            return Err(AtlasAffineError::TranslationLength {
                actual_len: translation.len(),
                expected_len: D,
            });
        }
        if center.len() != D {
            return Err(AtlasAffineError::CenterLength {
                actual_len: center.len(),
                expected_len: D,
            });
        }
        Ok(Self {
            matrix: matrix.to_vec(),
            translation: translation.to_vec(),
            center: center.to_vec(),
            _backend: std::marker::PhantomData,
        })
    }

    /// Panicking variant for parity with the legacy
    /// `AffineTransform::new` (which panics on matrix-shape
    /// conflict). Test-rewrite callers use this entry point for
    /// shape-equivalence assertions against the legacy oracle.
    pub fn construct(matrix: &[f32], translation: &[f32], center: &[f32]) -> Self {
        Self::try_new(matrix, translation, center).expect("AtlasAffineTransform::construct shape")
    }

    /// Construct an identity Atlas affine transform with optional center.
    pub fn identity(center: Option<&[f32]>) -> Self {
        let mut matrix = vec![0.0f32; D * D];
        for i in 0..D {
            matrix[i * (D + 1)] = 1.0;
        }
        let translation = vec![0.0f32; D];
        let center = center
            .map(|c| c.to_vec())
            .unwrap_or_else(|| vec![0.0f32; D]);
        Self::construct(&matrix, &translation, &center)
    }

    /// Get a borrowed slice over the `[D, D]` matrix (row-major).
    pub fn matrix(&self) -> &[f32] {
        &self.matrix
    }

    /// Get a borrowed slice over the `[D]` translation vector.
    pub fn translation(&self) -> &[f32] {
        &self.translation
    }

    /// Get a borrowed slice over the `[D]` center vector.
    pub fn center(&self) -> &[f32] {
        &self.center
    }

    /// Apply the transform to a rank-2 `[N, D]` points carrier.
    ///
    /// Sister to `AffineTransform::transform_points`. Computes
    /// `T(x) = A(x - c) + c + t` per point on the host slice and
    /// re-wraps the result in the Atlas-typed image carrier.
    /// `points` is always rank-2 with shape `[N, D]` (last axis width =
    /// outer `D`), and the output is also rank-2 — matching the legacy
    /// `Tensor<B, 2>::transform_points` contract verbatim (legacy never
    /// varies by per-rank generic for the points carrier).
    pub fn transform_points<BB>(
        &self,
        points: &Image<f32, BB, 2>,
    ) -> Result<Image<f32, MoiraiBackend, 2>, AtlasAffineError>
    where
        BB: ComputeBackend,
        BB::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let (point_values, point_shape_2d) = tensor_ops::extract_image_slice(points)
            .map_err(|e| AtlasAffineError::Extract(e.to_string()))?;
        if point_shape_2d[1] != D {
            return Err(AtlasAffineError::Construct(format!(
                "atlas affine transform_points expected last-axis width {D}, got {:?}",
                point_shape_2d
            )));
        }
        let n = point_shape_2d[0];
        let mut out_values = vec![0.0f32; n * D];
        for i in 0..n {
            for k in 0..D {
                let mut acc = 0.0f32;
                for j in 0..D {
                    let centered_j = point_values[i * D + j] - self.center[j];
                    acc += self.matrix[k * D + j] * centered_j;
                }
                out_values[i * D + k] = acc + self.center[k] + self.translation[k];
            }
        }
        Image::<f32, MoiraiBackend, 2>::from_flat(
            out_values,
            point_shape_2d,
            Point::<2>::origin(),
            Spacing::<2>::uniform(1.0),
            Direction::<2>::identity(),
        )
        .map_err(|e| AtlasAffineError::Construct(e.to_string()))
    }
}

// ── Row-major rotation-matrix builders (host math) ────────────────────────

/// Row-major `[D·D]` rotation matrix from Euler angles (radians). 3D uses the
/// ZYX composition `R = R_z(γ)·R_y(β)·R_x(α)` (`angles = [α, β, γ]`), 2D uses a
/// single angle, 1D/4D are identity — matching the Burn
/// `RigidTransform::build_rotation_matrix` host formulation.
fn euler_rotation_matrix<const D: usize>(angles: &[f32]) -> Vec<f32> {
    if D == 3 {
        let (cx, sx) = (angles[0].cos(), angles[0].sin());
        let (cy, sy) = (angles[1].cos(), angles[1].sin());
        let (cz, sz) = (angles[2].cos(), angles[2].sin());
        vec![
            cz * cy,
            cz * sy * sx - sz * cx,
            cz * sy * cx + sz * sx,
            sz * cy,
            sz * sy * sx + cz * cx,
            sz * sy * cx - cz * sx,
            -sy,
            cy * sx,
            cy * cx,
        ]
    } else if D == 2 {
        let (c, s) = (angles[0].cos(), angles[0].sin());
        vec![c, -s, s, c]
    } else {
        let mut m = vec![0.0f32; D * D];
        for i in 0..D {
            m[i * (D + 1)] = 1.0;
        }
        m
    }
}

/// Row-major `[9]` rotation matrix from a quaternion `[x, y, z, w]`, normalised
/// with the same `1e-12` guard as the Burn `VersorRigid3DTransform`; products
/// accumulate in `f64` to mirror that path's intermediate precision.
fn quaternion_rotation_matrix(quat: &[f32]) -> Vec<f32> {
    const QUAT_NORM_GUARD: f32 = 1e-12;
    let norm = quat[0]
        .mul_add(
            quat[0],
            quat[1].mul_add(quat[1], quat[2].mul_add(quat[2], quat[3] * quat[3])),
        )
        .sqrt()
        + QUAT_NORM_GUARD;
    let x = (quat[0] / norm) as f64;
    let y = (quat[1] / norm) as f64;
    let z = (quat[2] / norm) as f64;
    let w = (quat[3] / norm) as f64;

    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    let (xw, yw, zw) = (x * w, y * w, z * w);

    [
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy - zw),
        2.0 * (xz + yw),
        2.0 * (xy + zw),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz - xw),
        2.0 * (xz - yw),
        2.0 * (yz + xw),
        1.0 - 2.0 * (xx + yy),
    ]
    .into_iter()
    .map(|v| v as f32)
    .collect()
}

// ── Specialization constructors (SSOT: one affine type) ───────────────────

impl<B: ComputeBackend, const D: usize> AtlasAffineTransform<B, D> {
    /// Native sister of `TranslationTransform`: `T(x) = x + t` (identity matrix,
    /// zero center).
    pub fn from_translation(translation: &[f32]) -> Self {
        let mut matrix = vec![0.0f32; D * D];
        for i in 0..D {
            matrix[i * (D + 1)] = 1.0;
        }
        Self::construct(&matrix, translation, &vec![0.0f32; D])
    }

    /// Native sister of `ScaleTransform`: `T(x) = S(x − c) + c` (diagonal scale
    /// matrix, zero translation).
    pub fn from_scale(scale: &[f32], center: &[f32]) -> Self {
        let mut matrix = vec![0.0f32; D * D];
        for i in 0..D {
            matrix[i * (D + 1)] = scale[i];
        }
        Self::construct(&matrix, &vec![0.0f32; D], center)
    }

    /// Native sister of `RigidTransform`: `T(x) = R(x − c) + c + t` with `R`
    /// built from Euler `rotation` angles (radians; see `euler_rotation_matrix`).
    pub fn from_euler_rigid(translation: &[f32], rotation: &[f32], center: &[f32]) -> Self {
        let matrix = euler_rotation_matrix::<D>(rotation);
        Self::construct(&matrix, translation, center)
    }
}

impl<B: ComputeBackend> AtlasAffineTransform<B, 3> {
    /// Native sister of `VersorRigid3DTransform`: `T(x) = R(x − c) + c + t` with
    /// `R` built from the `quaternion` `[x, y, z, w]` (see
    /// `quaternion_rotation_matrix`).
    pub fn from_versor(translation: &[f32], quaternion: &[f32], center: &[f32]) -> Self {
        let matrix = quaternion_rotation_matrix(quaternion);
        Self::construct(&matrix, translation, center)
    }
}

#[cfg(test)]
#[path = "tests_atlas_ctors.rs"]
mod tests_atlas_ctors;
