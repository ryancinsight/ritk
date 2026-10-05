use super::parsed::NrrdKeyValueRecord;
use super::{copy_string, reserve_entry, NrrdHeader, NrrdHeaderError};

pub(super) fn insert_key_value(
    header: &mut NrrdHeader,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), NrrdHeaderError> {
    if key.is_empty() {
        return Err(NrrdHeaderError::EmptyKey { line_number });
    }
    reserve_entry(
        header.fields.len(),
        header.key_value_records.len(),
        header.comments.len(),
    )?;
    let key = unescape_key_value(key, line_number)?;
    let value = unescape_key_value(value, line_number)?;
    if !header.key_values.contains_key(&key) {
        header
            .key_values
            .try_reserve(1)
            .map_err(|source| NrrdHeaderError::Allocation {
                operation: "key/value table",
                source,
            })?;
    }
    header
        .key_value_records
        .try_reserve(1)
        .map_err(|source| NrrdHeaderError::Allocation {
            operation: "key/value record table",
            source,
        })?;
    let record_key = copy_string(&key, "key/value record key")?;
    let record_value = copy_string(&value, "key/value record value")?;
    header
        .key_value_records
        .push(NrrdKeyValueRecord::new(record_key, record_value));
    header.key_values.insert(key, value);
    Ok(())
}

fn unescape_key_value(value: &str, line_number: usize) -> Result<String, NrrdHeaderError> {
    let mut decoded = String::new();
    decoded
        .try_reserve_exact(value.len())
        .map_err(|source| NrrdHeaderError::Allocation {
            operation: "key/value string",
            source,
        })?;
    let bytes = value.as_bytes();
    let mut index = 0_usize;
    while let Some(byte) = bytes.get(index).copied() {
        if byte != b'\\' {
            decoded.push(char::from(byte));
            index += 1;
            continue;
        }
        index += 1;
        match bytes.get(index).copied() {
            Some(b'n') => decoded.push('\n'),
            Some(b'\\') => decoded.push('\\'),
            Some(_) | None => return Err(NrrdHeaderError::InvalidKeyValueEscape { line_number }),
        }
        index += 1;
    }
    Ok(decoded)
}
