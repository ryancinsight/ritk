//! Native scalar samples from the three mutually exclusive PS3.5 §8.2 pixel elements.
mod affine;
mod fragment;
mod jpeg2000;
mod sample;
pub use sample::StoredScalar;

use crate::dataset::{Dataset, DicomFile, PrimitiveValue, VR};
use crate::{tags, DicomTag, PixelLayout, PixelSignedness, TransferSyntaxKind};
use anyhow::{ensure, Context, Result};
use bytes::Bytes;

const PIXELS: DicomTag = DicomTag::new(0x7fe0, 0x0010);
const FLOAT: DicomTag = DicomTag::new(0x7fe0, 0x0008);
const DOUBLE: DicomTag = DicomTag::new(0x7fe0, 0x0009);
const URL: DicomTag = DicomTag::new(0x0028, 0x7fe0);

/// Pixel representation chosen from the payload tag, independently of its container width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarEncoding {
    Integer,
    Float32,
    Float64,
}

impl ScalarEncoding {
    /// Resolve pixel source exclusivity and the floating-point image pixel modules.
    pub fn from_dataset(dataset: &Dataset) -> Result<Self> {
        let present = |tag| dataset.get(tag).is_some();
        ensure!(
            [PIXELS, FLOAT, DOUBLE, URL]
                .into_iter()
                .filter(|&t| present(t))
                .count()
                <= 1,
            "multiple DICOM pixel sources in the same data set"
        );
        ensure!(
            !present(URL),
            "referenced Pixel Data Provider URL requires an external pixel source"
        );
        for (tag, vr, width, encoding) in [
            (FLOAT, VR::OF, 32, Self::Float32),
            (DOUBLE, VR::OD, 64, Self::Float64),
        ] {
            if let Some(field) = dataset.get(tag) {
                ensure!(
                    field.vr() == vr
                        && dataset.element(tags::BITS_ALLOCATED)?.to_num::<u16>()? == width,
                    "invalid DICOM floating pixel VR or BitsAllocated"
                );
                ensure!(
                    ![
                        tags::BITS_STORED,
                        DicomTag::new(0x0028, 0x0102),
                        tags::PIXEL_REPRESENTATION
                    ]
                    .into_iter()
                    .any(present),
                    "integer pixel attributes present for floating Pixel Data"
                );
                ensure!(
                    dataset.element(tags::SAMPLES_PER_PIXEL)?.to_num::<u16>()? == 1,
                    "floating pixel data requires a scalar layout"
                );
                return Ok(encoding);
            }
        }
        Ok(Self::Integer)
    }

    /// Effective sample precision used by the decoder. OF/OD have no BitsStored attribute.
    pub fn precision(self, dataset: &Dataset) -> Result<u16> {
        match self {
            Self::Float32 => Ok(32),
            Self::Float64 => Ok(64),
            Self::Integer => dataset
                .element(tags::BITS_STORED)
                .context("missing required BitsStored")?
                .to_num::<u16>(),
        }
    }

    const fn tag(self) -> DicomTag {
        match self {
            Self::Integer => PIXELS,
            Self::Float32 => FLOAT,
            Self::Float64 => DOUBLE,
        }
    }
}

/// Shared canonical little-endian samples, including exact values wider than a display float.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeScalarPixels {
    data: Bytes,
    encoding: ScalarEncoding,
    allocated: u16,
    stored: u16,
    signed: bool,
    rescale: [f64; 2],
    frame_rescale: Vec<[f64; 2]>,
}

impl NativeScalarPixels {
    /// Read the native value field without copying the parser's bulk byte buffer.
    pub fn from_object(
        obj: &DicomFile,
        encoding: ScalarEncoding,
        allocated: u16,
        stored: u16,
        representation: PixelSignedness,
    ) -> Result<Self> {
        let ts = TransferSyntaxKind::from_uid(obj.meta().transfer_syntax());
        ensure!(
            (!ts.is_encapsulated()
                || (ts.has_native_frame_layout() && encoding == ScalarEncoding::Integer))
                && !matches!(ts, TransferSyntaxKind::Unknown(_)),
            "unsupported native scalar transfer syntax"
        );
        ensure!(
            (allocated == 1 || allocated.is_multiple_of(8)) && stored > 0 && stored <= allocated,
            "invalid native DICOM sample width"
        );
        let field = obj.element(encoding.tag())?;
        let data = if ts.has_native_frame_layout() {
            fragment::collect_byte_aligned(obj, &ts, allocated)?
        } else {
            match field.value().primitive() {
                Some(PrimitiveValue::Bytes(bytes)) => bytes.clone(),
                _ => Bytes::from(field.to_bytes()?.into_owned()),
            }
        };
        Ok(Self {
            data,
            encoding,
            allocated,
            stored,
            signed: representation.is_signed(),
            rescale: Self::modality_rescale(obj),
            frame_rescale: Vec::new(),
        })
    }

    fn modality_rescale(obj: &DicomFile) -> [f64; 2] {
        [
            obj.get(DicomTag::new(0x0028, 0x1053))
                .and_then(|e| e.to_num::<f64>().ok())
                .unwrap_or(1.0),
            obj.get(DicomTag::new(0x0028, 0x1052))
                .and_then(|e| e.to_num::<f64>().ok())
                .unwrap_or(0.0),
        ]
    }

    /// Retain enhanced per-frame modality transforms in encoded frame order.
    pub fn set_frame_rescale(&mut self, transforms: Vec<[f64; 2]>) {
        self.frame_rescale = transforms;
    }

    /// Resolve a window offset from the acquired sample and its modality transform.
    pub fn window_offset(&self, index: usize, frame: usize, center: f64) -> Option<f64> {
        let [slope, intercept] = self
            .frame_rescale
            .get(frame)
            .copied()
            .unwrap_or(self.rescale);
        Some(self.get(index)?.window_offset(slope, intercept, center))
    }

    /// Get a source value without rounding through the viewer's rendering buffer.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<StoredScalar> {
        if self.allocated == 1 {
            let bit = (self.data.get(index / 8)? >> (index % 8)) & 1;
            return Some(if self.signed {
                StoredScalar::Signed(-i64::from(bit))
            } else {
                StoredScalar::Unsigned(u64::from(bit))
            });
        }
        let width = usize::from(self.allocated / 8);
        let start = index.checked_mul(width)?;
        let word = self.data.get(start..start.checked_add(width)?)?;
        Some(match self.encoding {
            ScalarEncoding::Float32 => {
                StoredScalar::Float(f32::from_le_bytes(word.try_into().ok()?))
            }
            ScalarEncoding::Float64 => {
                StoredScalar::Double(f64::from_le_bytes(word.try_into().ok()?))
            }
            ScalarEncoding::Integer => StoredScalar::integer(word, self.stored, self.signed),
        })
    }

    pub(crate) fn decode(
        &self,
        index: usize,
        frames: usize,
        layout: PixelLayout,
    ) -> Result<Vec<f32>> {
        layout.validate_rescale_parameters()?;
        let count = layout.samples_per_frame()?;
        let total = frames
            .checked_mul(count)
            .context("native DICOM sample count overflow")?;
        let bits = total
            .checked_mul(usize::from(self.allocated))
            .context("native DICOM bit count overflow")?;
        ensure!(
            self.data.len() >= bits.div_ceil(8),
            "truncated native DICOM pixel payload"
        );
        let start = index
            .checked_mul(count)
            .context("native frame offset overflow")?;
        let end = start
            .checked_add(count)
            .context("native frame end overflow")?;
        ensure!(end <= total, "frame index outside NumberOfFrames");
        (start..end)
            .map(|i| {
                self.get(i)
                    .context("native sample outside value field")
                    .map(|v| v.display_value(layout))
            })
            .collect()
    }
}
