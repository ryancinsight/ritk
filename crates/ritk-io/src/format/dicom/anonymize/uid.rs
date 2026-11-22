//! Deterministic DICOM UID remapping.

use sha2::{Digest, Sha256};

/// Produce a deterministic DICOM-conformant UID from `original` and `salt`.
///
/// # Algorithm
/// SHA-256 over `original || "||" || salt`, the first 19 bytes of the digest
/// are converted to decimal digits (each byte → 2 or 3 decimal digits),
/// producing at most 57 decimal characters. The result is prefixed with
/// `"2.25."` (ISO/IEC 9834-8 UUID arc), yielding a UID ≤ 64 characters.
///
/// # Invariants
/// - Same `(original, salt)` → same output (pure function).
/// - Output matches `^2\.25\.[0-9]+$` and is ≤ 64 characters.
/// - Original UID cannot be recovered without the salt.
/// - Collision probability is O(1/2^152) (19 bytes of SHA-256 entropy).
pub(crate) fn generate_uid_from_hash(original: &str, salt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(original.as_bytes());
    hasher.update(b"||");
    hasher.update(salt.as_bytes());
    let digest = hasher.finalize();

    // Convert first 19 bytes of SHA-256 digest to decimal string.
    // Each byte [0,255] → 2 or 3 decimal digits → max 57 characters.
    // With "2.25." prefix (5 chars), total ≤ 62 characters < 64 max.
    let decimal: String = digest[..19]
        .iter()
        .flat_map(|b| {
            let d = u32::from(*b);
            if d < 10 {
                vec![b'0', b'0', b'0' + (d as u8)]
            } else if d < 100 {
                let tens = (d / 10) as u8;
                let ones = (d % 10) as u8;
                vec![b'0', b'0' + tens, b'0' + ones]
            } else {
                let hundreds = (d / 100) as u8;
                let rem = (d % 100) as u8;
                let tens = rem / 10;
                let ones = rem % 10;
                vec![b'0' + hundreds, b'0' + tens, b'0' + ones]
            }
        })
        .map(|b| b as char)
        .collect();

    // Strip leading zeros from the decimal portion to avoid UID components
    // with leading zeros (DICOM forbids leading-zero components after the root).
    let decimal_stripped = decimal.trim_start_matches('0');
    let uid_body = if decimal_stripped.is_empty() {
        "0"
    } else {
        decimal_stripped
    };
    let uid = format!("2.25.{uid_body}");

    debug_assert!(
        uid.len() <= 64,
        "Generated UID exceeds 64-char DICOM limit: {} (len={})",
        uid,
        uid.len()
    );
    debug_assert!(
        uid.chars().all(|c| c.is_ascii_digit() || c == '.'),
        "Generated UID contains invalid characters: {uid}"
    );
    debug_assert!(
        !uid.contains(".00"),
        "Generated UID has leading-zero component: {uid}"
    );

    uid
}
