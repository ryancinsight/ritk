use anyhow::Result;
use ritk_spatial::Direction;
use std::io::{self, Write};

/// Flatten a 3×3 direction-cosine matrix to the row-major layout the space
/// directions builder consumes.
pub(super) fn direction_row_major(direction: &Direction<3>) -> [f64; 9] {
    let d = direction.0;
    [
        d[(0, 0)],
        d[(0, 1)],
        d[(0, 2)],
        d[(1, 0)],
        d[(1, 1)],
        d[(1, 2)],
        d[(2, 0)],
        d[(2, 1)],
        d[(2, 2)],
    ]
}

/// Write `values` as little-endian IEEE 754 f32.
///
/// On little-endian targets the slice reinterprets to bytes with no copy; a
/// per-element `write_all` loop is far slower across millions of voxels.
pub(super) fn write_float_payload(writer: &mut impl Write, values: &[f32]) -> Result<()> {
    #[cfg(target_endian = "little")]
    writer.write_all(bytemuck::cast_slice(values))?;
    #[cfg(target_endian = "big")]
    {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        writer.write_all(&bytes)?;
    }
    Ok(())
}

pub(crate) struct HeaderBuffer {
    bytes: Vec<u8>,
    exceeded_limit: bool,
}

impl HeaderBuffer {
    pub(crate) fn new() -> Self {
        Self {
            bytes: Vec::new(),
            exceeded_limit: false,
        }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) const fn exceeded_limit(&self) -> bool {
        self.exceeded_limit
    }
}

impl Write for HeaderBuffer {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        let maximum_bytes = crate::reader::MAX_HEADER_BYTES;
        let remaining = maximum_bytes - self.bytes.len();
        if input.len() > remaining {
            self.bytes
                .try_reserve(remaining)
                .map_err(io::Error::other)?;
            self.bytes.extend_from_slice(&input[..remaining]);
            self.exceeded_limit = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "NRRD output header exceeds the parser limit",
            ));
        }
        self.bytes
            .try_reserve(input.len())
            .map_err(io::Error::other)?;
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) fn format_nrrd_vector(vector: [f64; 3]) -> String {
    format!("({},{},{})", vector[0], vector[1], vector[2])
}
