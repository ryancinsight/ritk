//! Lossless single-file NIfTI document transport.

use crate::header::{checked_spatial_pixdim, qfac_from_pixdim, HeaderVersion, NiftiHeader};
use crate::reader::GZIP_MAGIC;
use crate::writer::is_gzip_path;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::borrow::Cow;
use std::fmt;
use std::fs;
use std::io::{self, Read, Write};
use std::ops::Range;
use std::path::Path;

const MAX_DOCUMENT_BYTES: u64 = 1 << 30;

/// NIfTI header version carried by a [`NiftiDocument`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NiftiVersion {
    /// NIfTI-1.1, with a 348-byte header.
    One,
    /// NIfTI-2, with a 540-byte header.
    Two,
}

/// Relationship between the NIfTI spatial transforms present in a document.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SpatialFormRelation {
    /// Neither transform is active; only voxel sizes are defined.
    None,
    /// Only the scanner-oriented quaternion transform is active.
    QformOnly,
    /// Only the general affine transform is active.
    SformOnly,
    /// Both transforms are active and have the same handedness.
    CompatibleHandedness,
    /// Both transforms are active but disagree about handedness.
    HandednessConflict,
}

/// Parsed header semantics needed to inspect a lossless NIfTI document.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct NiftiDocumentHeader {
    /// Header layout version.
    pub version: NiftiVersion,
    /// Rank followed by the seven declared dimensions.
    pub dimensions: [usize; 8],
    /// NIfTI datatype code.
    pub datatype_code: i16,
    /// Stored bits per sample.
    pub bits_per_sample: u16,
    /// Sampling intervals, including the qform handedness value at index zero.
    pub pixel_dimensions: [f64; 8],
    /// Start of the sample payload in the uncompressed stream.
    pub voxel_offset: usize,
    /// Quaternion transform code.
    pub qform_code: i32,
    /// General affine transform code.
    pub sform_code: i32,
    /// Packed spatial and temporal unit codes.
    pub xyzt_units: i32,
    /// Relationship between active spatial transforms.
    pub spatial_forms: SpatialFormRelation,
}

/// Typed failure while reading or writing a complete NIfTI document.
#[derive(Debug)]
#[non_exhaustive]
pub enum NiftiDocumentError {
    /// The compressed stream is invalid or exceeds the decoded-byte limit.
    Compression(io::Error),
    /// The NIfTI header is invalid.
    Header(anyhow::Error),
    /// The declared sample range is invalid.
    Payload(anyhow::Error),
    /// Active qform fields are invalid.
    SpatialForms(anyhow::Error),
    /// The active sform has no finite handedness.
    DegenerateSform,
    /// The expanded document exceeds the bounded allocation.
    DecodedSizeLimit(u64),
    /// File-system I/O failed.
    Io(io::Error),
}

impl fmt::Display for NiftiDocumentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compression(error) => write!(formatter, "invalid gzip stream: {error}"),
            Self::Header(error) => write!(formatter, "invalid NIfTI header: {error}"),
            Self::Payload(error) => write!(formatter, "invalid NIfTI payload: {error}"),
            Self::SpatialForms(error) => write!(formatter, "invalid NIfTI qform: {error}"),
            Self::DegenerateSform => {
                formatter.write_str("active NIfTI sform must have a finite nonzero determinant")
            }
            Self::DecodedSizeLimit(limit) => {
                write!(formatter, "decoded NIfTI document exceeds {limit} bytes")
            }
            Self::Io(error) => write!(formatter, "NIfTI document I/O failed: {error}"),
        }
    }
}

impl std::error::Error for NiftiDocumentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Compression(error) | Self::Io(error) => Some(error),
            Self::Header(error) | Self::Payload(error) | Self::SpatialForms(error) => {
                Some(error.root_cause())
            }
            Self::DegenerateSform | Self::DecodedSizeLimit(_) => None,
        }
    }
}

/// A validated, byte-preserving single-file NIfTI document.
#[derive(Clone, Debug, PartialEq)]
pub struct NiftiDocument {
    header: NiftiDocumentHeader,
    bytes: Vec<u8>,
    samples: Range<usize>,
}

impl NiftiDocument {
    /// Parse uncompressed `.nii` bytes or a complete `.nii.gz` stream.
    ///
    /// Gzip input is drained through its trailer, so checksum failures are
    /// reported instead of accepting only the declared sample prefix.
    pub fn from_bytes(encoded: &[u8]) -> Result<Self, NiftiDocumentError> {
        let bytes = if encoded.starts_with(&GZIP_MAGIC) {
            decode_gzip(encoded)?
        } else {
            encoded.to_vec()
        };
        let parsed = NiftiHeader::parse(&bytes).map_err(NiftiDocumentError::Header)?;
        let samples = parsed
            .volume_byte_range(bytes.len())
            .map_err(NiftiDocumentError::Payload)?;
        let spatial_forms = classify_forms(&parsed)?;
        let bits_per_sample = u16::try_from(parsed.datatype.byte_width() * 8)
            .expect("invariant: supported NIfTI sample widths fit u16");
        let version = match parsed.version {
            HeaderVersion::One => NiftiVersion::One,
            HeaderVersion::Two => NiftiVersion::Two,
        };
        let header = NiftiDocumentHeader {
            version,
            dimensions: parsed.dim,
            datatype_code: parsed.datatype.code(),
            bits_per_sample,
            pixel_dimensions: parsed.pixdim,
            voxel_offset: parsed.vox_offset,
            qform_code: parsed.qform_code,
            sform_code: parsed.sform_code,
            xyzt_units: parsed.xyzt_units,
            spatial_forms,
        };
        Ok(Self {
            header,
            bytes,
            samples,
        })
    }

    /// Read and validate one `.nii` or `.nii.gz` document.
    pub fn read(path: impl AsRef<Path>) -> Result<Self, NiftiDocumentError> {
        let encoded = fs::read(path).map_err(NiftiDocumentError::Io)?;
        Self::from_bytes(&encoded)
    }

    /// Return the parsed header view.
    #[must_use]
    pub const fn header(&self) -> &NiftiDocumentHeader {
        &self.header
    }

    /// Return the exact stored sample bytes in file byte order.
    #[must_use]
    pub fn sample_bytes(&self) -> &[u8] {
        &self.bytes[self.samples.clone()]
    }

    /// Return the complete uncompressed single-file stream.
    #[must_use]
    pub fn uncompressed_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Write the document without changing any uncompressed byte.
    ///
    /// The `.gz` suffix selects gzip framing. Compression completes before the
    /// destination is opened, so compression errors cannot alter an existing
    /// output.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), NiftiDocumentError> {
        let path = path.as_ref();
        let encoded = if is_gzip_path(path) {
            Cow::Owned(encode_gzip(&self.bytes)?)
        } else {
            Cow::Borrowed(self.bytes.as_slice())
        };
        fs::write(path, encoded.as_ref()).map_err(NiftiDocumentError::Io)
    }
}

/// Validate and transcode a single-file NIfTI document between `.nii` and
/// `.nii.gz` framing.
pub fn transcode_nifti_document(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
) -> Result<(), NiftiDocumentError> {
    NiftiDocument::read(source)?.write(destination)
}

fn classify_forms(header: &NiftiHeader) -> Result<SpatialFormRelation, NiftiDocumentError> {
    match (header.qform_code > 0, header.sform_code > 0) {
        (false, false) => Ok(SpatialFormRelation::None),
        (true, false) => Ok(SpatialFormRelation::QformOnly),
        (false, true) => Ok(SpatialFormRelation::SformOnly),
        (true, true) => {
            checked_spatial_pixdim(header.pixdim)
                .and_then(|_| qfac_from_pixdim(header.pixdim[0]))
                .map_err(NiftiDocumentError::SpatialForms)?;
            let [a, b, c] = [header.srow_x, header.srow_y, header.srow_z];
            let determinant = a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
            if !determinant.is_finite() || determinant == 0.0 {
                return Err(NiftiDocumentError::DegenerateSform);
            }
            let qform_is_left_handed = header.pixdim[0] == -1.0;
            if qform_is_left_handed == determinant.is_sign_negative() {
                Ok(SpatialFormRelation::CompatibleHandedness)
            } else {
                Ok(SpatialFormRelation::HandednessConflict)
            }
        }
    }
}

fn decode_gzip(encoded: &[u8]) -> Result<Vec<u8>, NiftiDocumentError> {
    let mut decoded = Vec::new();
    GzDecoder::new(encoded)
        .take(MAX_DOCUMENT_BYTES + 1)
        .read_to_end(&mut decoded)
        .map_err(NiftiDocumentError::Compression)?;
    if u64::try_from(decoded.len()).expect("invariant: usize fits u64") > MAX_DOCUMENT_BYTES {
        return Err(NiftiDocumentError::DecodedSizeLimit(MAX_DOCUMENT_BYTES));
    }
    Ok(decoded)
}

fn encode_gzip(bytes: &[u8]) -> Result<Vec<u8>, NiftiDocumentError> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    encoder
        .write_all(bytes)
        .map_err(NiftiDocumentError::Compression)?;
    encoder.finish().map_err(NiftiDocumentError::Compression)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{HeaderDims, HeaderSpatial, NiftiDatatype};
    use tempfile::tempdir;

    fn document_bytes(sform_x: f64) -> Vec<u8> {
        let mut header = NiftiHeader::new_volume(
            HeaderDims {
                nx: 2,
                ny: 1,
                nz: 1,
            },
            NiftiDatatype::Float32,
            HeaderSpatial {
                pixdim: [1.0; 8],
                srow_x: [sform_x, 0.0, 0.0, 0.0],
                srow_y: [0.0, 1.0, 0.0, 0.0],
                srow_z: [0.0, 0.0, 1.0, 0.0],
            },
        )
        .expect("valid test header");
        header.qform_code = 1;
        header.sform_code = 2;
        header.vox_offset = 368;
        let mut bytes = header.encode();
        bytes.extend_from_slice(&[1, 0, 0, 0]);
        bytes.extend_from_slice(&16_i32.to_le_bytes());
        bytes.extend_from_slice(&6_i32.to_le_bytes());
        bytes.extend_from_slice(&[11, 22, 33, 44, 55, 66, 77, 88]);
        bytes.extend_from_slice(&0x8000_0000_u32.to_le_bytes());
        bytes.extend_from_slice(&0x7fc0_1234_u32.to_le_bytes());
        bytes
    }

    #[test]
    fn document_round_trip_preserves_header_extension_and_sample_bits() {
        let bytes = document_bytes(1.0);
        let document = NiftiDocument::from_bytes(&bytes).expect("valid document");

        assert_eq!(document.sample_bytes(), &bytes[368..]);
        assert_eq!(document.header().qform_code, 1);
        assert_eq!(document.header().sform_code, 2);
        assert_eq!(document.header().bits_per_sample, 32);
        assert_eq!(
            document.header().spatial_forms,
            SpatialFormRelation::CompatibleHandedness
        );
        let directory = tempdir().expect("temporary directory is available");
        for name in ["roundtrip.nii", "roundtrip.nii.gz"] {
            let path = directory.path().join(name);
            document.write(&path).expect("document writes");
            let reread = NiftiDocument::read(path).expect("written document reads");
            assert_eq!(reread.uncompressed_bytes(), bytes);
        }
    }

    #[test]
    fn gzip_round_trip_drains_trailer_and_preserves_document() {
        let bytes = document_bytes(1.0);
        let encoded = encode_gzip(&bytes).expect("gzip encoding succeeds");

        let mut corrupt = encoded;
        let last = corrupt.last_mut().expect("gzip stream has a trailer");
        *last ^= 1;
        assert!(matches!(
            NiftiDocument::from_bytes(&corrupt),
            Err(NiftiDocumentError::Compression(_))
        ));
    }

    #[test]
    fn opposite_qform_and_sform_handedness_is_explicit() {
        let document = NiftiDocument::from_bytes(&document_bytes(-1.0))
            .expect("both valid forms remain representable");
        assert_eq!(
            document.header().spatial_forms,
            SpatialFormRelation::HandednessConflict
        );
    }

    #[test]
    fn invalid_source_does_not_change_existing_destination() {
        let directory = tempdir().expect("temporary directory is available");
        let source = directory.path().join("invalid.nii.gz");
        let destination = directory.path().join("existing.nii");
        fs::write(&source, [0x1f, 0x8b, 0, 0]).expect("invalid source fixture writes");
        fs::write(&destination, b"keep me").expect("destination fixture writes");

        assert!(transcode_nifti_document(&source, &destination).is_err());
        assert_eq!(
            fs::read(destination).expect("destination remains readable"),
            b"keep me"
        );
    }
}
