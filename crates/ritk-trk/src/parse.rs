#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::io::Read;

use crate::{TrkError, TrkHeader};

/// Magic bytes that identify a `.trk` file.
pub(crate) const TRK_MAGIC: &[u8; 6] = b"TRACK\0";

/// Size of the fixed header in bytes.
pub(crate) const TRK_HEADER_SIZE: usize = 1000;

pub(crate) fn parse_header(buf: &[u8; 1000]) -> TrkHeader {
    let dim = [
        i16_from_le(buf, 6),
        i16_from_le(buf, 8),
        i16_from_le(buf, 10),
    ];
    let voxel_size = [
        f32_from_le(buf, 12),
        f32_from_le(buf, 16),
        f32_from_le(buf, 20),
    ];
    let origin = [
        f32_from_le(buf, 24),
        f32_from_le(buf, 28),
        f32_from_le(buf, 32),
    ];
    let n_scalars = i16_from_le(buf, 36);
    let mut scalar_name = [0u8; 200];
    scalar_name.copy_from_slice(&buf[38..238]);
    let n_properties = i16_from_le(buf, 238);
    let mut property_name = [0u8; 200];
    property_name.copy_from_slice(&buf[240..440]);

    let mut vox_to_ras = [[0f32; 4]; 4];
    for (row, values) in vox_to_ras.iter_mut().enumerate() {
        for (col, value) in values.iter_mut().enumerate() {
            *value = f32_from_le(buf, 440 + (row * 4 + col) * 4);
        }
    }

    let mut voxel_order = [0u8; 4];
    voxel_order.copy_from_slice(&buf[948..952]);
    let image_orientation_patient = [
        f32_from_le(buf, 956),
        f32_from_le(buf, 960),
        f32_from_le(buf, 964),
        f32_from_le(buf, 968),
        f32_from_le(buf, 972),
        f32_from_le(buf, 976),
    ];
    let invert_x = buf[982];
    let invert_y = buf[983];
    let invert_z = buf[984];
    let swap_xy = buf[985];
    let swap_yz = buf[986];
    let swap_zx = buf[987];
    let n_count = i32_from_le(buf, 988);
    let version = i32_from_le(buf, 992);
    let hdr_size = i32_from_le(buf, 996);

    TrkHeader {
        dim,
        voxel_size,
        origin,
        n_scalars,
        scalar_name,
        n_properties,
        property_name,
        vox_to_ras,
        voxel_order,
        image_orientation_patient,
        invert_x,
        invert_y,
        invert_z,
        swap_xy,
        swap_yz,
        swap_zx,
        n_count,
        version,
        hdr_size,
    }
}

pub(crate) fn encode_header(header: &TrkHeader) -> Vec<u8> {
    let mut buf = vec![0u8; TRK_HEADER_SIZE];

    buf[..6].copy_from_slice(TRK_MAGIC);

    for (i, v) in header.dim.iter().enumerate() {
        buf[6 + i * 2..8 + i * 2].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in header.voxel_size.iter().enumerate() {
        buf[12 + i * 4..16 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in header.origin.iter().enumerate() {
        buf[24 + i * 4..28 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    buf[36..38].copy_from_slice(&header.n_scalars.to_le_bytes());
    buf[38..238].copy_from_slice(&header.scalar_name);
    buf[238..240].copy_from_slice(&header.n_properties.to_le_bytes());
    buf[240..440].copy_from_slice(&header.property_name);
    for row in 0..4 {
        for col in 0..4 {
            let off = 440 + (row * 4 + col) * 4;
            buf[off..off + 4].copy_from_slice(&header.vox_to_ras[row][col].to_le_bytes());
        }
    }
    buf[948..952].copy_from_slice(&header.voxel_order);
    for (i, v) in header.image_orientation_patient.iter().enumerate() {
        buf[956 + i * 4..960 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    buf[982] = header.invert_x;
    buf[983] = header.invert_y;
    buf[984] = header.invert_z;
    buf[985] = header.swap_xy;
    buf[986] = header.swap_yz;
    buf[987] = header.swap_zx;
    buf[988..992].copy_from_slice(&header.n_count.to_le_bytes());
    buf[992..996].copy_from_slice(&header.version.to_le_bytes());
    buf[996..1000].copy_from_slice(&header.hdr_size.to_le_bytes());

    buf
}

pub(crate) fn read_exact(
    reader: &mut impl Read,
    buf: &mut [u8],
    offset: &mut usize,
) -> Result<(), TrkError> {
    reader
        .read_exact(buf)
        .map_err(|_| TrkError::UnexpectedEof { offset: *offset })?;
    *offset += buf.len();
    Ok(())
}

pub(crate) fn i16_from_le(buf: &[u8], pos: usize) -> i16 {
    i16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap())
}

pub(crate) fn i32_from_le(buf: &[u8], pos: usize) -> i32 {
    i32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap())
}

pub(crate) fn f32_from_le(buf: &[u8], pos: usize) -> f32 {
    f32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap())
}

/// Apply a row-major `4×4` affine to a point `(x, y, z, 1)`.
pub(crate) fn apply_affine(affine: &[[f32; 4]; 4], x: f32, y: f32, z: f32) -> (f64, f64, f64) {
    let rx = affine[0][0] as f64 * x as f64
        + affine[0][1] as f64 * y as f64
        + affine[0][2] as f64 * z as f64
        + affine[0][3] as f64;
    let ry = affine[1][0] as f64 * x as f64
        + affine[1][1] as f64 * y as f64
        + affine[1][2] as f64 * z as f64
        + affine[1][3] as f64;
    let rz = affine[2][0] as f64 * x as f64
        + affine[2][1] as f64 * y as f64
        + affine[2][2] as f64 * z as f64
        + affine[2][3] as f64;
    (rx, ry, rz)
}

/// Compute the inverse of a `4×4` affine stored in row-major order.
///
/// Only the upper-left `3×3` and the translation column `3` are inverted;
/// the bottom row `[0, 0, 0, 1]` is assumed and preserved.
pub(crate) fn invert_affine(affine: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let a: [[f64; 4]; 4] = [
        [
            affine[0][0] as f64,
            affine[0][1] as f64,
            affine[0][2] as f64,
            affine[0][3] as f64,
        ],
        [
            affine[1][0] as f64,
            affine[1][1] as f64,
            affine[1][2] as f64,
            affine[1][3] as f64,
        ],
        [
            affine[2][0] as f64,
            affine[2][1] as f64,
            affine[2][2] as f64,
            affine[2][3] as f64,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let r = [
        [a[0][0], a[0][1], a[0][2]],
        [a[1][0], a[1][1], a[1][2]],
        [a[2][0], a[2][1], a[2][2]],
    ];

    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);

    if det.abs() < 1e-30 {
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
    }

    let inv_det = 1.0 / det;
    let inv_r = [
        [
            (r[1][1] * r[2][2] - r[1][2] * r[2][1]) * inv_det,
            (r[0][2] * r[2][1] - r[0][1] * r[2][2]) * inv_det,
            (r[0][1] * r[1][2] - r[0][2] * r[1][1]) * inv_det,
        ],
        [
            (r[1][2] * r[2][0] - r[1][0] * r[2][2]) * inv_det,
            (r[0][0] * r[2][2] - r[0][2] * r[2][0]) * inv_det,
            (r[0][2] * r[1][0] - r[0][0] * r[1][2]) * inv_det,
        ],
        [
            (r[1][0] * r[2][1] - r[1][1] * r[2][0]) * inv_det,
            (r[0][1] * r[2][0] - r[0][0] * r[2][1]) * inv_det,
            (r[0][0] * r[1][1] - r[0][1] * r[1][0]) * inv_det,
        ],
    ];

    let tx = a[0][3];
    let ty = a[1][3];
    let tz = a[2][3];
    let inv_tx = -(inv_r[0][0] * tx + inv_r[0][1] * ty + inv_r[0][2] * tz);
    let inv_ty = -(inv_r[1][0] * tx + inv_r[1][1] * ty + inv_r[1][2] * tz);
    let inv_tz = -(inv_r[2][0] * tx + inv_r[2][1] * ty + inv_r[2][2] * tz);

    [
        [
            inv_r[0][0] as f32,
            inv_r[0][1] as f32,
            inv_r[0][2] as f32,
            inv_tx as f32,
        ],
        [
            inv_r[1][0] as f32,
            inv_r[1][1] as f32,
            inv_r[1][2] as f32,
            inv_ty as f32,
        ],
        [
            inv_r[2][0] as f32,
            inv_r[2][1] as f32,
            inv_r[2][2] as f32,
            inv_tz as f32,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
