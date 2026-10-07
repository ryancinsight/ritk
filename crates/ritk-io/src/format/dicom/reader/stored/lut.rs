use dicom::core::{Tag, VR};
use dicom::object::DefaultDicomObject;
use ritk_codecs::PixelSignedness;
use ritk_image_io::{LutOutputBits, ModalityLookupTable};

use super::StoredDicomError;

pub(super) fn modality_lookup_table(
    object: &DefaultDicomObject,
    pixel_representation: PixelSignedness,
) -> Result<Option<(ModalityLookupTable, String)>, StoredDicomError> {
    let Ok(sequence) = object.element(Tag(0x0028, 0x3000)) else {
        return Ok(None);
    };
    let items = sequence
        .items()
        .ok_or(StoredDicomError::InvalidModalityLookupTable {
            field: "Modality LUT Sequence is not a sequence",
        })?;
    let [item] = items else {
        return Err(StoredDicomError::InvalidModalityLookupTable {
            field: "Modality LUT Sequence must contain one item",
        });
    };
    let lut_type = item.element(Tag(0x0028, 0x3004)).map_err(|_| {
        StoredDicomError::InvalidModalityLookupTable {
            field: "Modality LUT Type (0028,3004) is absent from the sequence item",
        }
    })?;
    if lut_type.vr() != VR::LO {
        return Err(StoredDicomError::InvalidModalityLookupTable {
            field: "Modality LUT Type (0028,3004) must use LO",
        });
    }
    let unit = lut_type
        .to_str()
        .map_err(|_| StoredDicomError::InvalidModalityLookupTable {
            field: "Modality LUT Type (0028,3004) is not text",
        })?
        .into_owned();
    if unit.trim().is_empty() {
        return Err(StoredDicomError::InvalidModalityLookupTable {
            field: "Modality LUT Type (0028,3004) is empty",
        });
    }
    let descriptor = item.element(Tag(0x0028, 0x3002)).map_err(|_| {
        StoredDicomError::InvalidModalityLookupTable {
            field: "LUT Descriptor (0028,3002) is absent",
        }
    })?;
    let expected_vr = match pixel_representation {
        PixelSignedness::Unsigned => VR::US,
        PixelSignedness::Signed => VR::SS,
    };
    if descriptor.vr() != expected_vr {
        return Err(StoredDicomError::InvalidModalityLookupTable {
            field: "LUT Descriptor VR disagrees with Pixel Representation",
        });
    }
    let values = descriptor.to_multi_int::<i64>().map_err(|_| {
        StoredDicomError::InvalidModalityLookupTable {
            field: "LUT Descriptor must contain three integers",
        }
    })?;
    let [raw_entry_count, raw_first_mapped_value, raw_output_bits] = values.as_slice() else {
        return Err(StoredDicomError::InvalidModalityLookupTable {
            field: "LUT Descriptor must contain three values",
        });
    };
    let (entry_count, first_mapped_value, output_bits) = match descriptor.vr() {
        VR::US => (
            u16::try_from(*raw_entry_count).map_err(|_| {
                StoredDicomError::InvalidModalityLookupTable {
                    field: "LUT entry count is outside the unsigned descriptor range",
                }
            })?,
            i64::from(u16::try_from(*raw_first_mapped_value).map_err(|_| {
                StoredDicomError::InvalidModalityLookupTable {
                    field: "first mapped value is outside the unsigned descriptor range",
                }
            })?),
            u16::try_from(*raw_output_bits).map_err(|_| {
                StoredDicomError::InvalidModalityLookupTable {
                    field: "LUT output precision is outside the unsigned descriptor range",
                }
            })?,
        ),
        VR::SS => {
            let raw_entry_count = i16::try_from(*raw_entry_count).map_err(|_| {
                StoredDicomError::InvalidModalityLookupTable {
                    field: "LUT entry count is outside the signed descriptor range",
                }
            })?;
            let raw_first_mapped_value = i16::try_from(*raw_first_mapped_value).map_err(|_| {
                StoredDicomError::InvalidModalityLookupTable {
                    field: "first mapped value is outside the signed descriptor range",
                }
            })?;
            let raw_output_bits = i16::try_from(*raw_output_bits).map_err(|_| {
                StoredDicomError::InvalidModalityLookupTable {
                    field: "LUT output precision is outside the signed descriptor range",
                }
            })?;
            (
                u16::from_le_bytes(raw_entry_count.to_le_bytes()),
                i64::from(raw_first_mapped_value),
                u16::from_le_bytes(raw_output_bits.to_le_bytes()),
            )
        }
        _ => {
            return Err(StoredDicomError::InvalidModalityLookupTable {
                field: "LUT Descriptor must use US or SS",
            });
        }
    };
    let entry_count = if entry_count == 0 {
        65_536
    } else {
        usize::from(entry_count)
    };
    let output_bits = match output_bits {
        8 => LutOutputBits::Eight,
        16 => LutOutputBits::Sixteen,
        _ => {
            return Err(StoredDicomError::InvalidModalityLookupTable {
                field: "LUT output precision must be 8 or 16 bits",
            });
        }
    };
    let data = item.element(Tag(0x0028, 0x3006)).map_err(|_| {
        StoredDicomError::InvalidModalityLookupTable {
            field: "LUT Data (0028,3006) is absent",
        }
    })?;
    if !matches!(data.vr(), VR::US | VR::OW) {
        return Err(StoredDicomError::InvalidModalityLookupTable {
            field: "LUT Data must use US or OW",
        });
    }
    let words =
        data.to_multi_int::<u16>()
            .map_err(|_| StoredDicomError::InvalidModalityLookupTable {
                field: "LUT Data contains a value outside unsigned 16-bit range",
            })?;
    let entries = match output_bits {
        LutOutputBits::Sixteen if words.len() == entry_count => words,
        LutOutputBits::Eight if words.len() == entry_count => {
            if words.iter().any(|word| *word > u16::from(u8::MAX)) {
                return Err(StoredDicomError::InvalidModalityLookupTable {
                    field: "8-bit LUT entry exceeds its declared range",
                });
            }
            words
        }
        LutOutputBits::Eight
            if words.len() == entry_count.div_ceil(2) && words.len() < entry_count =>
        {
            let mut entries = Vec::new();
            entries
                .try_reserve_exact(entry_count)
                .map_err(StoredDicomError::Allocation)?;
            for word in words {
                let [low, high] = word.to_le_bytes();
                entries.push(u16::from(low));
                if entries.len() < entry_count {
                    entries.push(u16::from(high));
                } else if high != 0 {
                    return Err(StoredDicomError::InvalidModalityLookupTable {
                        field: "odd 8-bit LUT entry count has nonzero final padding",
                    });
                }
            }
            entries
        }
        _ => {
            return Err(StoredDicomError::InvalidModalityLookupTable {
                field: "LUT Data length disagrees with LUT Descriptor",
            });
        }
    };
    ModalityLookupTable::new(first_mapped_value, entries.into_boxed_slice(), output_bits)
        .map(|table| Some((table, unit)))
        .map_err(StoredDicomError::Calibration)
}
