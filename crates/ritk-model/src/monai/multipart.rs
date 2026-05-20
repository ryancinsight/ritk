//! Minimal RFC 2046 multipart body parser for MONAI Label Server responses.
//!
//! # RFC 2046 §5.1 boundary grammar (simplified)
//!
//! ```text
//! multipart-body  := preamble 1*encapsulation close-delimiter
//! encapsulation   := delimiter CRLF body-part CRLF
//! delimiter       := "--" boundary
//! close-delimiter := "--" boundary "--"
//! body-part       := MIME-part-headers CRLF body-bytes
//! ```
//!
//! This parser tolerates both `\r\n` (CRLF) and `\n` (LF) line endings.

// ── Public API ────────────────────────────────────────────────────────────────

/// Split a `multipart/form-data` body into `(header_bytes, part_body_bytes)` pairs.
///
/// `boundary` must be the raw boundary string extracted from the `Content-Type` header
/// (without the leading `--`).  Returns an empty `Vec` for malformed or empty input.
///
/// # RFC 2046 semantics
///
/// Each part's body is the bytes between the double-CRLF header separator and the
/// start of the next `--<boundary>` delimiter, with trailing CRLF stripped.
pub(crate) fn split_multipart<'a>(
    body: &'a [u8],
    boundary: &[u8],
) -> Vec<(&'a [u8], &'a [u8])> {
    // Build the `--<boundary>` delimiter.
    let mut delim: Vec<u8> = Vec::with_capacity(2 + boundary.len());
    delim.extend_from_slice(b"--");
    delim.extend_from_slice(boundary);

    // Split the whole body by `--<boundary>`.
    // segments[0] = preamble (skip); segments[last] starts with "--" (close delimiter, skip).
    let segments = split_bytes(body, &delim);
    if segments.len() < 2 {
        return Vec::new();
    }

    let mut parts = Vec::new();
    for seg in &segments[1..] {
        // Strip the leading CRLF that follows the delimiter on its own line.
        let seg = strip_leading_crlf(seg);
        // Close-delimiter segment starts with "--"; we are done.
        if seg.starts_with(b"--") {
            break;
        }
        // Split headers and body at the first blank line (\r\n\r\n or \n\n).
        let (hdr, body_raw) = split_at_double_crlf(seg);
        // Strip trailing CRLF from the part body (it belongs to the next delimiter line).
        let body_trimmed = trim_trailing_crlf(body_raw);
        parts.push((hdr, body_trimmed));
    }
    parts
}

/// Split a part's bytes into `(header_bytes, body_bytes)` at the first blank line.
///
/// Blank line separator: `\r\n\r\n` (preferred) or `\n\n` (fallback).
/// If no separator is found, the whole slice is treated as headers with an empty body.
pub(crate) fn split_at_double_crlf(data: &[u8]) -> (&[u8], &[u8]) {
    if let Some(pos) = find_seq(data, b"\r\n\r\n") {
        return (&data[..pos], &data[pos + 4..]);
    }
    if let Some(pos) = find_seq(data, b"\n\n") {
        return (&data[..pos], &data[pos + 2..]);
    }
    (data, b"")
}

/// Extract the `name` attribute value from a Content-Disposition header byte slice.
///
/// Matches `name="<value>"` (quoted) and `name=<value>` (unquoted), case-insensitive.
/// Returns `None` if the attribute is absent or the bytes are not valid UTF-8.
pub(crate) fn extract_part_name(headers: &[u8]) -> Option<String> {
    let h = std::str::from_utf8(headers).ok()?.to_ascii_lowercase();
    let name_pos = h.find("name=")?;
    let rest = &h[name_pos + 5..];
    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        Some(stripped[..end].to_owned())
    } else {
        let end = rest
            .find(|c: char| matches!(c, ';' | '\r' | '\n' | ' '))
            .unwrap_or(rest.len());
        Some(rest[..end].to_owned())
    }
}

// ── Byte utilities ────────────────────────────────────────────────────────────

/// Split `data` by `sep`, returning a `Vec` of non-overlapping sub-slices.
///
/// Semantics match `str::split` applied to byte slices: the separator is not
/// included in any returned slice, and zero-length slices are retained.
pub(crate) fn split_bytes<'a>(data: &'a [u8], sep: &[u8]) -> Vec<&'a [u8]> {
    if sep.is_empty() {
        return vec![data];
    }
    let mut result = Vec::new();
    let mut start = 0usize;
    loop {
        match find_seq(&data[start..], sep) {
            Some(rel) => {
                result.push(&data[start..start + rel]);
                start += rel + sep.len();
            }
            None => {
                result.push(&data[start..]);
                break;
            }
        }
    }
    result
}

/// Return the index of the first occurrence of `needle` in `haystack`, or `None`.
pub(crate) fn find_seq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Strip a single leading `\r\n` or `\n` from `data`.
fn strip_leading_crlf(data: &[u8]) -> &[u8] {
    if data.starts_with(b"\r\n") {
        &data[2..]
    } else if data.starts_with(b"\n") {
        &data[1..]
    } else {
        data
    }
}

/// Remove all trailing `\r` and `\n` bytes from `data`.
fn trim_trailing_crlf(data: &[u8]) -> &[u8] {
    let mut end = data.len();
    while end > 0 && matches!(data[end - 1], b'\r' | b'\n') {
        end -= 1;
    }
    &data[..end]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a well-formed multipart body for unit tests.
    ///
    /// `parts` = slice of `(name, content_type, body_bytes)`.
    pub(crate) fn build_multipart(boundary: &str, parts: &[(&str, &str, &[u8])]) -> Vec<u8> {
        let mut body = Vec::new();
        for (name, ct, data) in parts {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{name}\"\r\nContent-Type: {ct}\r\n\r\n"
                )
                .as_bytes(),
            );
            body.extend_from_slice(data);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        body
    }

    #[test]
    fn split_bytes_by_known_sep_produces_correct_slices() {
        let data = b"alpha--sep--beta--sep--gamma";
        let parts = split_bytes(data, b"--sep--");
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], b"alpha");
        assert_eq!(parts[1], b"beta");
        assert_eq!(parts[2], b"gamma");
    }

    #[test]
    fn split_at_double_crlf_separates_headers_and_body() {
        let data = b"Header: value\r\n\r\nbody content";
        let (hdr, body) = split_at_double_crlf(data);
        assert_eq!(hdr, b"Header: value");
        assert_eq!(body, b"body content");
    }

    #[test]
    fn extract_part_name_quoted_returns_name() {
        let hdr = b"Content-Disposition: form-data; name=\"label\"; filename=\"label.nii.gz\"";
        assert_eq!(extract_part_name(hdr).as_deref(), Some("label"));
    }

    #[test]
    fn extract_part_name_unquoted_returns_name() {
        let hdr = b"Content-Disposition: form-data; name=params";
        assert_eq!(extract_part_name(hdr).as_deref(), Some("params"));
    }

    #[test]
    fn split_multipart_two_parts_returns_both() {
        let label_data: &[u8] = b"\x01\x02\x03NIFTI";
        let params_data: &[u8] = br#"{"k":"v"}"#;
        let boundary = "Bound42";
        let body = build_multipart(
            boundary,
            &[
                ("label", "application/octet-stream", label_data),
                ("params", "application/json", params_data),
            ],
        );
        let parts = split_multipart(&body, boundary.as_bytes());
        assert_eq!(parts.len(), 2, "must yield exactly 2 parts");

        let (h0, b0) = parts[0];
        assert_eq!(extract_part_name(h0).as_deref(), Some("label"));
        assert_eq!(b0, label_data);

        let (h1, b1) = parts[1];
        assert_eq!(extract_part_name(h1).as_deref(), Some("params"));
        assert_eq!(b1, params_data);
    }

    #[test]
    fn split_multipart_empty_body_returns_empty() {
        let parts = split_multipart(b"", b"boundary");
        assert!(parts.is_empty());
    }
}
