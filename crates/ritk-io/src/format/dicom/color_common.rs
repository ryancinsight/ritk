use anyhow::{Context, Result};

use ritk_dicom::dataset::Dataset;
use ritk_dicom::dataset::DicomTag;

pub(super) const RGB_CHANNELS: usize = 3;

pub(super) fn read_required_unsigned(obj: &Dataset, tag: DicomTag, name: &str) -> Result<u16> {
    let attribute = obj.element(tag).with_context(|| {
        format!(
            "DICOM missing required {name} ({:04X},{:04X})",
            tag.group, tag.element
        )
    })?;
    let cardinality = attribute.value().cardinality();
    if cardinality != 1 {
        anyhow::bail!(
            "DICOM required {name} ({:04X},{:04X}) has cardinality={cardinality}; expected exactly 1",
            tag.group,
            tag.element
        );
    }
    attribute.to_num::<u16>().with_context(|| {
        format!(
            "DICOM required {name} ({:04X},{:04X}) is not an unsigned scalar",
            tag.group, tag.element
        )
    })
}

pub(super) fn read_required<T>(
    obj: &ritk_dicom::dataset::DicomFile,
    tag: DicomTag,
    name: &str,
) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    obj.element(tag)
        .with_context(|| format!("{name} absent"))?
        .to_str()
        .with_context(|| format!("{name} unreadable"))?
        .trim()
        .parse::<T>()
        .with_context(|| format!("{name} invalid"))
}

pub(super) fn read_optional<T: std::str::FromStr>(
    obj: &ritk_dicom::dataset::DicomFile,
    tag: DicomTag,
) -> Option<T> {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
}

pub(super) fn required_string(
    obj: &ritk_dicom::dataset::DicomFile,
    tag: DicomTag,
    name: &str,
) -> Result<String> {
    Ok(obj
        .element(tag)
        .with_context(|| format!("{name} absent"))?
        .to_str()
        .with_context(|| format!("{name} unreadable"))?
        .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_dicom::dataset::Dataset;
    use ritk_dicom::dataset::{DataElement, PrimitiveValue, VR};

    const BITS_STORED: DicomTag = DicomTag::new(0x0028, 0x0101);

    #[test]
    fn required_unsigned_rejects_missing_and_malformed_values() {
        let mut object = Dataset::new();
        let missing = read_required_unsigned(&object, BITS_STORED, "BitsStored")
            .expect_err("missing BitsStored must be rejected");
        assert!(
            missing.to_string().contains("missing required BitsStored"),
            "expected missing attribute error, got {missing:#}"
        );

        object.put(DataElement::new(
            BITS_STORED,
            VR::LO,
            PrimitiveValue::from("twelve"),
        ));
        let malformed = read_required_unsigned(&object, BITS_STORED, "BitsStored")
            .expect_err("malformed BitsStored must be rejected");
        assert!(
            malformed.to_string().contains("is not an unsigned scalar"),
            "expected malformed attribute error, got {malformed:#}"
        );

        object.put(DataElement::new(
            BITS_STORED,
            VR::US,
            PrimitiveValue::U16([12_u16, 16][..].into()),
        ));
        let multi_value = read_required_unsigned(&object, BITS_STORED, "BitsStored")
            .expect_err("multi-valued BitsStored must be rejected");
        assert!(
            multi_value.to_string().contains("cardinality=2"),
            "expected exact cardinality error, got {multi_value:#}"
        );

        object.put(DataElement::new(
            BITS_STORED,
            VR::US,
            PrimitiveValue::from(12_u16),
        ));
        assert_eq!(
            read_required_unsigned(&object, BITS_STORED, "BitsStored")
                .expect("valid BitsStored must decode"),
            12
        );
    }
}
