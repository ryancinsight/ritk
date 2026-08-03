#![forbid(unsafe_code)]
#![deny(missing_docs)]

use gaia::Polyline;

/// The fixed 1000-byte header of a `.trk` file.
///
/// All multi-byte integer fields are little-endian.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrkHeader {
    /// Voxel grid dimensions `[nx, ny, nz]`.
    pub dim: [i16; 3],

    /// Voxel size in millimetres `[dx, dy, dz]`.
    pub voxel_size: [f32; 3],

    /// Origin in millimetres `[ox, oy, oz]` (typically zero in modern files
    /// because the origin is folded into the affine).
    pub origin: [f32; 3],

    /// Number of per-point scalar values stored after the position triplet.
    pub n_scalars: i16,

    /// Space-separated null-terminated names for each per-point scalar.
    pub scalar_name: [u8; 200],

    /// Number of per-streamline property values.
    pub n_properties: i16,

    /// Space-separated null-terminated names for each per-streamline property.
    pub property_name: [u8; 200],

    /// Row-major `4×4` affine that transforms voxel indices `[i, j, k, 1]`
    /// to physical RAS+mm coordinates.
    pub vox_to_ras: [[f32; 4]; 4],

    /// Voxel ordering convention, e.g. `b"LPS\0"`.
    pub voxel_order: [u8; 4],

    /// DICOM Image Orientation Patient (6 floats).
    pub image_orientation_patient: [f32; 6],

    /// Axis inversion flags (`0` or `1`).
    pub invert_x: u8,
    /// Axis inversion flags (`0` or `1`).
    pub invert_y: u8,
    /// Axis inversion flags (`0` or `1`).
    pub invert_z: u8,

    /// Axis swap flags (`0` or `1`).
    pub swap_xy: u8,
    /// Axis swap flags (`0` or `1`).
    pub swap_yz: u8,
    /// Axis swap flags (`0` or `1`).
    pub swap_zx: u8,

    /// Total number of streamlines declared in the file.
    pub n_count: i32,

    /// Format version (current: `2`).
    pub version: i32,

    /// Header size in bytes (must be `1000`).
    pub hdr_size: i32,
}

impl Default for TrkHeader {
    fn default() -> Self {
        Self {
            dim: [0, 0, 0],
            voxel_size: [1.0, 1.0, 1.0],
            origin: [0.0, 0.0, 0.0],
            n_scalars: 0,
            scalar_name: [0u8; 200],
            n_properties: 0,
            property_name: [0u8; 200],
            vox_to_ras: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            voxel_order: [b'R', b'A', b'S', 0],
            image_orientation_patient: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            invert_x: 0,
            invert_y: 0,
            invert_z: 0,
            swap_xy: 0,
            swap_yz: 0,
            swap_zx: 0,
            n_count: 0,
            version: 2,
            hdr_size: 1000,
        }
    }
}

/// The full contents of a `.trk` file.
///
/// Streamlines are stored in physical RAS+mm coordinates after the header
/// affine has been applied.
#[derive(Debug, Clone)]
pub struct TrkTractogram {
    /// Fixed 1000-byte header metadata.
    pub header: TrkHeader,

    /// Streamlines in file order, each a valid Gaia polyline.
    pub streamlines: Vec<Polyline<f64>>,

    /// Per-point scalar values stored inline with the streamline data.
    ///
    /// `scalars[i]` is a flat array of `n_points × n_scalars` f32 values
    /// for streamline `i`. Empty when `header.n_scalars == 0`.
    pub scalars: Vec<Box<[f32]>>,

    /// Per-streamline properties stored after each streamline.
    ///
    /// `properties[i]` is a slice of `n_properties` f32 values for
    /// streamline `i`. Empty when `header.n_properties == 0`.
    pub properties: Vec<Box<[f32]>>,
}
