//! XML parsing helpers shared by the VTI ASCII and binary-appended readers.

/// Return the opening tag string for the first occurrence of `<tag ...>` or
/// `<tag>` in `s`, including the closing `>`.
pub(crate) fn find_tag(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let start = s.find(&open)?;
    let end = s[start..].find('>')? + 1;
    Some(s[start..start + end].to_string())
}

/// Return the substring from the first `<tag` to the matching `</tag>` (inclusive).
pub(crate) fn find_section(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let start = s.find(&open)?;
    let end_offset = s[start..].find(&close)? + close.len();
    Some(s[start..start + end_offset].to_string())
}

/// Parse the `name="value"` attribute from an XML tag string.
pub(crate) fn attr_val(tag: &str, name: &str) -> Option<String> {
    let dq = char::from(34u8); // "
    let mut pat = name.to_string();
    pat.push(char::from(61u8)); // =
    pat.push(dq);
    let start = tag.find(&pat)? + pat.len();
    let rest = &tag[start..];
    let end = rest.find(dq)?;
    Some(rest[..end].to_string())
}

/// Parse space/newline-delimited values from a string slice.
///
/// `T` must implement [`std::str::FromStr`]; tokens that fail to parse are
/// silently dropped.  Supports `f32`, `f64`, `i64`, and any other type with a
/// `FromStr` impl.
pub(crate) fn parse_floats<T>(s: &str) -> Vec<T>
where
    T: std::str::FromStr,
{
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// Parse space/newline-delimited i64 values from a string slice.
pub(crate) fn parse_i64s(s: &str) -> Vec<i64> {
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}
