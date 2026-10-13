//! Spatial metadata transforms between MGH RAS header fields and RITK images.
//!
//! This module is the single owner of two reconciliations between the MGH
//! header and RITK. FreeSurfer orders the header's `spacing` and `Mdc` fields
//! by the x, y, z voxel axes, while RITK orders image axes
//! `[depth, row, col] = [z, y, x]` (`docs/architecture.md` §7–§9); and the
//! header states its volume center and direction cosines in RAS, while the
//! stored model is LPS-millimetre (`docs/architecture.md` §5). Keeping both
//! reconciliations here is what stops the reader, the writer, and their tests
//! from disagreeing about them.
//!
//! The `Mdc` field is stored **transposed** — FreeSurfer's own header
//! descriptor names it the transpose of the direction-cosine matrix and its
//! consumers build `MdcD = Mdcᵀ · D` — so the field's three consecutive
//! triples are the direction-cosine **columns**. No transpose is owed; only
//! the frame changes.

use ritk_spatial::{Direction, InvalidSpacing, Point, Spacing, Vector};

/// Whether the RAS (Right-Anterior-Superior) spatial metadata in the MGH
/// header is valid and should be used to derive image geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RasValidity {
    /// RAS fields are valid — use them to compute origin, spacing, direction.
    Valid,
    /// RAS fields are absent or unreliable — fall back to identity geometry.
    Synthetic,
}

/// The MGH header's RAS block, in header axis order `[x, y, z]`.
///
/// The header stores `D = diag(d_x, d_y, d_z)` and `Mdc = [x_ras, y_ras, z_ras]`
/// (`docs/book/mgh_format.md`; crate module docs). RITK stores the same
/// geometry in `[depth, row, col] = [z, y, x]` axis order, the file-axis to
/// internal-axis reversal NIfTI, NRRD, and MetaImage also apply. Every field
/// here is in header order; conversion to and from RITK order happens in
/// [`MghRasBlock::into_image_geometry`] and [`ras_block_from_geometry`].
pub(crate) struct MghRasBlock {
    /// Voxel spacing in header order `[Δx, Δy, Δz]`.
    spacing_xyz: [f32; 3],
    /// Direction-cosine columns in header order `[x_ras, y_ras, z_ras]`.
    mdc_columns_xyz: [[f32; 3]; 3],
    /// RAS coordinate of the volume center.
    c_ras: [f32; 3],
}

/// Image geometry in RITK `[depth, row, col]` axis order.
pub(crate) struct ImageGeometry {
    /// Voxel spacing in RITK axis order `[Δdepth, Δrow, Δcol]`.
    pub(crate) spacing: Spacing<3>,
    /// Direction-cosine columns in RITK axis order `[depth, row, col]`.
    pub(crate) direction: Direction<3>,
    /// Physical coordinate of voxel index zero.
    pub(crate) origin: Point<3>,
}

impl MghRasBlock {
    /// Assemble a block from header fields already in header order.
    pub(crate) const fn new(
        spacing_xyz: [f32; 3],
        mdc_columns_xyz: [[f32; 3]; 3],
        c_ras: [f32; 3],
    ) -> Self {
        Self {
            spacing_xyz,
            mdc_columns_xyz,
            c_ras,
        }
    }

    /// Voxel spacing in header order, ready to serialize.
    pub(crate) const fn spacing_xyz(&self) -> [f32; 3] {
        self.spacing_xyz
    }

    /// Direction-cosine columns in header order, ready to serialize.
    pub(crate) const fn mdc_columns_xyz(&self) -> [[f32; 3]; 3] {
        self.mdc_columns_xyz
    }

    /// RAS coordinate of the volume center, ready to serialize.
    pub(crate) const fn c_ras(&self) -> [f32; 3] {
        self.c_ras
    }

    /// Derive RITK `[depth, row, col]` geometry from this header block.
    ///
    /// `dims_xyz` is the header's `[width, height, depth]`. With
    /// [`RasValidity::Synthetic`] the block is ignored and the result is unit
    /// spacing, identity direction, and a zero origin.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidSpacing`] when a header spacing component is not
    /// strictly positive and finite. A malformed file is a recoverable error
    /// rather than a panic.
    pub(crate) fn into_image_geometry(
        self,
        ras_validity: RasValidity,
        dims_xyz: [usize; 3],
    ) -> Result<ImageGeometry, InvalidSpacing> {
        if ras_validity == RasValidity::Synthetic {
            return Ok(ImageGeometry {
                spacing: Spacing::new([1.0, 1.0, 1.0]),
                direction: Direction::identity(),
                origin: Point::new([0.0, 0.0, 0.0]),
            });
        }

        let spacing_xyz = Spacing::try_new(widen_components(self.spacing_xyz))?;
        let direction_ras = direction_from_columns(self.mdc_columns_xyz);
        let origin_ras = Vector::new(widen_components(self.c_ras))
            - centered_half_offset(direction_ras, spacing_xyz, dims_xyz);
        // The center and every direction column are RAS; the stored model is
        // LPS, so both lose the sign of their x and y components.
        let direction_lps = flip_ras_to_lps_columns(direction_ras);

        Ok(ImageGeometry {
            // RITK axis order [depth, row, col] is the header order reversed.
            spacing: reverse_axes(spacing_xyz),
            direction: reverse_columns(direction_lps),
            origin: Point::new(flip_ras_to_lps_components(origin_ras.to_array())),
        })
    }
}

/// Project RITK `[depth, row, col]` geometry onto the MGH header RAS block.
///
/// `shape_zyx` is the RITK image shape `[depth, row, col]`. The returned block
/// carries the header's `[x, y, z]` order, including the `c_ras` volume center
/// that `origin` implies.
pub(crate) fn ras_block_from_geometry(
    shape_zyx: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
) -> MghRasBlock {
    // The header axis order is the RITK axis order reversed.
    let dims_xyz = [shape_zyx[2], shape_zyx[1], shape_zyx[0]];
    let spacing_xyz = reverse_axes(spacing);
    let direction_lps = reverse_columns(direction);
    let c_ras_lps =
        Vector::new(origin.to_array()) + centered_half_offset(direction_lps, spacing_xyz, dims_xyz);
    // The frame flip is its own inverse, so the same negation that the reader
    // applies returns the stored model's LPS center and columns to RAS.
    let direction_ras = flip_ras_to_lps_columns(direction_lps);

    MghRasBlock {
        spacing_xyz: narrow_spacing(spacing_xyz),
        mdc_columns_xyz: header_columns(direction_ras),
        c_ras: flip_ras_to_lps_components(c_ras_lps.to_array()).map(|v| v as f32),
    }
}

/// Widen a header `[f32; 3]` triple to `[f64; 3]` (lossless).
fn widen_components(v: [f32; 3]) -> [f64; 3] {
    v.map(f64::from)
}

/// Narrow RITK spacing to the header's `[f32; 3]`.
fn narrow_spacing(spacing: Spacing<3>) -> [f32; 3] {
    spacing.to_array().map(|v| v as f32)
}

/// Build RITK `Direction` from header-order direction-cosine columns.
fn direction_from_columns(columns: [[f32; 3]; 3]) -> Direction<3> {
    Direction::from_columns(columns.map(|column| Vector::new(widen_components(column))))
}

/// Reverse a spacing triple between header `[x, y, z]` and RITK
/// `[depth, row, col]`.
///
/// The input is already validated strictly positive, so reversing components
/// preserves the [`Spacing`] invariant.
fn reverse_axes(spacing: Spacing<3>) -> Spacing<3> {
    let [x, y, z] = spacing.to_array();
    Spacing::new([z, y, x])
}

/// Reverse direction columns between header `[x, y, z]` and RITK
/// `[depth, row, col]`.
fn reverse_columns(direction: Direction<3>) -> Direction<3> {
    let [x, y, z] = direction.axis_directions_array();
    Direction::from_columns([z, y, x])
}

/// Move a component triple between the header's RAS frame and the stored
/// model's LPS frame.
///
/// RAS and LPS share the superior axis and differ in the sign of the right/left
/// and anterior/posterior axes, so this is the negation `diag(-1, -1, 1)` and
/// is its own inverse. `ritk-nifti::spatial` negates the same two rows of a
/// NIfTI sform (`docs/architecture.md` §5, §21).
fn flip_ras_to_lps_components(components: [f64; 3]) -> [f64; 3] {
    [-components[0], -components[1], components[2]]
}

/// Apply [`flip_ras_to_lps_components`] to every direction-cosine column.
///
/// Negating two rows of an orthonormal matrix leaves it orthonormal, so the
/// result is still a valid [`Direction`]; it reverses the handedness of the
/// basis, which is exactly the RAS-to-LPS relation.
fn flip_ras_to_lps_columns(direction: Direction<3>) -> Direction<3> {
    Direction::from_columns(
        direction
            .axis_directions_array()
            .map(|column| Vector::new(flip_ras_to_lps_components(column.to_array()))),
    )
}

/// Extract header-order direction-cosine columns from RITK `Direction`.
fn header_columns(direction: Direction<3>) -> [[f32; 3]; 3] {
    direction
        .axis_directions_array()
        .map(|column| column.to_array().map(|v| v as f32))
}

/// Half-offset from voxel zero to the volume center, in header `[x, y, z]` order.
///
/// `Mdc · D · h` with `h = [(nx−1)/2, (ny−1)/2, (nz−1)/2]ᵀ`.
fn centered_half_offset(
    direction: Direction<3>,
    spacing: Spacing<3>,
    dims_xyz: [usize; 3],
) -> Vector<3> {
    let half_dim = Vector::new([
        (dims_xyz[0] as f64 - 1.0) / 2.0,
        (dims_xyz[1] as f64 - 1.0) / 2.0,
        (dims_xyz[2] as f64 - 1.0) / 2.0,
    ]);
    let scaled_half = Vector::new([
        spacing[0] * half_dim[0],
        spacing[1] * half_dim[1],
        spacing[2] * half_dim[2],
    ]);

    direction * scaled_half
}
