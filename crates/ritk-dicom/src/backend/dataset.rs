//! RITK dataset and frame operations at the provider boundary.

use std::borrow::Cow;
use std::path::Path;

use anyhow::{bail, Context, Result};
use bytes::Bytes;

use super::{
    BoundedDicomParseBackend, DecodeFrameRequest, DecodedFrame, DicomParseBackend,
    DicomWriteBackend, EncapsulatedFrameSource, NativeCodecBackend, PixelDecodeBackend,
};
use crate::dataset::{DicomFile, Value};
use crate::{tags, DicomTag, ParseBudget, PixelLayout, TransferSyntaxKind};

#[derive(Debug, Clone, Copy, Default)]
pub struct NativeBackend;

const PIXELS: DicomTag = DicomTag::new(0x7fe0, 0x0010);
const FRAMES: DicomTag = DicomTag::new(0x0028, 0x0008);
const EXTENDED_OFFSETS: DicomTag = DicomTag::new(0x7fe0, 0x0001);
const EXTENDED_LENGTHS: DicomTag = DicomTag::new(0x7fe0, 0x0002);

impl DicomParseBackend for NativeBackend {
    type Object = DicomFile;

    fn parse_file(path: &Path) -> Result<Self::Object> {
        DicomFile::open(path)
    }

    fn parse_bytes(data: &[u8]) -> Result<Self::Object> {
        Self::parse_bytes_with_budget(data, &ParseBudget::DEFAULT)
    }
}

impl BoundedDicomParseBackend for NativeBackend {
    fn parse_file_with_budget(path: &Path, budget: &ParseBudget) -> Result<Self::Object> {
        DicomFile::open_with_budget(path, budget)
    }

    fn parse_bytes_with_budget(data: &[u8], budget: &ParseBudget) -> Result<Self::Object> {
        super::parse_budget::check_input_bytes(data, budget)?;
        DicomFile::parse(Bytes::copy_from_slice(data), budget)
    }
}

impl DicomWriteBackend for NativeBackend {
    type Object = DicomFile;

    fn write_file(path: &Path, object: &Self::Object) -> Result<()> {
        object.write_to_file(path)
    }

    fn write_bytes(object: &Self::Object) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        object.write_all(&mut buffer)?;
        Ok(buffer)
    }
}

impl EncapsulatedFrameSource for DicomFile {
    fn encapsulated_frame(&self, frame_index: u32) -> Result<Cow<'_, [u8]>> {
        let Value::PixelSequence(pixel) = self.element(PIXELS)?.value() else {
            bail!("DICOM Pixel Data is not encapsulated");
        };
        let frames = self
            .get(FRAMES)
            .map_or(Ok(1), |field| field.to_num::<usize>())?;
        let extended = |tag| -> Result<Option<Vec<u64>>> {
            let Some(element) = self.get(tag) else {
                return Ok(None);
            };
            let bytes = element.value().to_bytes()?;
            if !bytes.len().is_multiple_of(8) {
                bail!("DICOM extended offset table {tag} has a partial entry");
            }
            Ok(Some(
                bytes
                    .chunks_exact(8)
                    .map(|word| {
                        consus_core::read_integer(word, consus_core::ByteOrder::LittleEndian)
                            .expect("eight-byte extended offset table entry")
                    })
                    .collect(),
            ))
        };
        let offsets = extended(EXTENDED_OFFSETS)?;
        let lengths = extended(EXTENDED_LENGTHS)?;
        let extension = match (offsets.as_deref(), lengths.as_deref()) {
            (Some(offsets), Some(lengths)) => Some((offsets, lengths)),
            (None, None) => None,
            _ => bail!("DICOM Extended Offset Table and its Lengths must occur together"),
        };
        pixel.frame(usize::try_from(frame_index)?, frames, extension)
    }
}

impl PixelDecodeBackend<DicomFile> for NativeBackend {
    fn decode_frame(object: &DicomFile, request: DecodeFrameRequest) -> Result<DecodedFrame> {
        let found = TransferSyntaxKind::from_uid(object.meta().transfer_syntax());
        if found != request.transfer_syntax {
            bail!(
                "DICOM requested transfer syntax {} differs from the file syntax {}",
                request.transfer_syntax.uid(),
                found.uid()
            );
        }
        validate_pixel_description(object, request.layout)?;
        if found.is_encapsulated() {
            return NativeCodecBackend::decode_frame(object, request);
        }
        if matches!(found, TransferSyntaxKind::Unknown(_)) {
            bail!("unsupported DICOM pixel transfer syntax {}", found.uid());
        }
        let bytes = object.element(PIXELS)?.value().to_bytes()?;
        let frame_bytes = request.layout.bytes_per_frame()?;
        let index = usize::try_from(request.frame_index)?;
        if let Some(frames) = object.get(FRAMES)
            && index >= frames.to_num::<usize>()?
        {
            bail!("DICOM frame {index} exceeds NumberOfFrames");
        }
        let start = index
            .checked_mul(frame_bytes)
            .context("native DICOM frame offset overflow")?;
        let end = start
            .checked_add(frame_bytes)
            .context("native DICOM frame end overflow")?;
        let frame = bytes.get(start..end).with_context(|| format!(
            "frame {index} out of range for native Pixel Data byte length {} and frame byte length {frame_bytes}",
            bytes.len()
        ))?;
        let pixels = crate::decode_native_pixel_bytes_checked(frame, request.layout)?;
        Ok(DecodedFrame { pixels })
    }
}

fn validate_pixel_description(object: &DicomFile, layout: PixelLayout) -> Result<()> {
    layout.bytes_per_sample()?;
    for (tag, name, declared) in [
        (tags::BITS_ALLOCATED, "BitsAllocated", layout.bits_allocated),
        (tags::BITS_STORED, "BitsStored", layout.bits_stored),
    ] {
        if let Some(element) = object.get(tag) {
            let stored = element.to_num::<u16>()?;
            if stored != declared {
                bail!("DICOM {name}={stored} does not match frame layout {name}={declared}");
            }
        }
    }
    if let Some(element) = object.get(DicomTag::new(0x0028, 0x0102)) {
        let high_bit = element.to_num::<u16>()?;
        let expected = layout.bits_stored - 1;
        if high_bit != expected {
            bail!(
                "DICOM HighBit={high_bit} does not equal BitsStored-1={expected} for BitsStored={}",
                layout.bits_stored
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
