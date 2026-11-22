//! Deterministic DICOM UID remapping.

use sha2::{Digest, Sha256};

/// Treat the first 128 bits of SHA-256(original || "||" || salt) as UUIDv8
/// bytes, with the RFC 9562 version and variant bits set. The separator is
/// outside the alphabet of a DICOM UID. The mapping leaves 122 digest bits.
/// The DICOM PS3.5 B.2 suffix is the decimal value of those bytes in network
/// order. An unsigned 128-bit integer requires at most 39 decimal digits, so
/// the result has at most 44 characters including the root.
pub(crate) fn generate_uid_from_hash(original: &str, salt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(original.as_bytes());
    hasher.update(b"||");
    hasher.update(salt.as_bytes());
    let digest = hasher.finalize();

    let mut bytes = [0; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x80;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!("2.25.{}", u128::from_be_bytes(bytes))
}
