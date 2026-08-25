use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parse the NRRD header into a key-value map without decoding the payload.
///
/// This is the shared header-parsing path used by [`super::read_nrrd`],
/// [`super::read_nrrd_series`], and [`super::read_nrrd_gradient_scheme`]. Keys are
/// lowercased for case-insensitive lookup. Comment lines (starting with `#`)
/// are skipped.
///
/// # Errors
///
/// Returns an error when the file cannot be opened, when the magic line is
/// absent or invalid, or when header lines cannot be read.
pub fn read_nrrd_header_map<P: AsRef<Path>>(path: P) -> Result<HashMap<String, String>> {
    let path = path.as_ref();
    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open NRRD file {:?}", path))?;
    let mut reader = BufReader::new(file);
    parse_nrrd_header_map_from_reader(&mut reader)
}

/// Parse the NRRD header from an already-opened reader.
pub(super) fn parse_nrrd_header_map_from_reader<R: BufRead>(
    reader: &mut R,
) -> Result<HashMap<String, String>> {
    let mut magic = String::new();
    reader
        .read_line(&mut magic)
        .context("Failed to read NRRD magic line")?;
    if !magic.trim_start().starts_with("NRRD") {
        return Err(anyhow!(
            "Not a valid NRRD file: magic line does not start with 'NRRD' (got '{}')",
            magic.trim()
        ));
    }

    let mut headers: HashMap<String, String> = HashMap::new();
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .context("Error reading NRRD header line")?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if trimmed.starts_with('#') {
            continue;
        }
        if let Some(colon_pos) = trimmed.find(':') {
            let key = trimmed[..colon_pos].trim().to_lowercase();
            let rest = &trimmed[colon_pos + 1..];
            // NRRD distinguishes header fields (`key: value`) from key/value
            // pairs (`key:=value`). Both are stored under the bare key, with
            // the `=` consumed, so a caller reading a custom field does not
            // have to strip it and cannot accidentally keep it as data.
            let value = rest.strip_prefix('=').unwrap_or(rest).trim().to_string();
            headers.insert(key, value);
        }
    }
    Ok(headers)
}
