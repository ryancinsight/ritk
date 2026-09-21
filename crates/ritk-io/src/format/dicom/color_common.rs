use anyhow::{Context, Result};
use dicom::core::value::DicomValueType;
use dicom::core::Tag;
use dicom::object::{DicomAttribute, DicomObject};

pub(super) const RGB_CHANNELS: usize = 3;

pub(super) fn read_required_unsigned<O>(obj: &O, tag: Tag, name: &str) -> Result<u16>
where
    O: DicomObject,
{
    let attribute = obj.attr(tag).with_context(|| {
        format!(
            "DICOM missing required {name} ({:04X},{:04X})",
            tag.0, tag.1
        )
    })?;
    let cardinality = attribute.cardinality();
    if cardinality != 1 {
        anyhow::bail!(
            "DICOM required {name} ({:04X},{:04X}) has cardinality={cardinality}; expected exactly 1",
            tag.0,
            tag.1
        );
    }
    attribute.to_u16().with_context(|| {
        format!(
            "DICOM required {name} ({:04X},{:04X}) is not an unsigned scalar",
            tag.0, tag.1
        )
    })
}

pub(super) fn read_required<T>(
    obj: &dicom::object::DefaultDicomObject,
    tag: Tag,
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
    obj: &dicom::object::DefaultDicomObject,
    tag: Tag,
) -> Option<T> {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
}

pub(super) fn required_string(
    obj: &dicom::object::DefaultDicomObject,
    tag: Tag,
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
    use dicom::core::{DataElement, PrimitiveValue, VR};
    use dicom::object::InMemDicomObject;

    const BITS_STORED: Tag = Tag(0x0028, 0x0101);

    #[test]
    fn required_unsigned_rejects_missing_and_malformed_values() {
        let mut object = InMemDicomObject::new_empty();
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
