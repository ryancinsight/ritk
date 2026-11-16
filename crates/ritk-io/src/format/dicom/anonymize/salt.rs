//! UID salts drawn per export.

use super::{AnonymizationProfile, AnonymizeOptions};

/// Bytes of operating-system randomness in a fresh salt. The salt keys a
/// SHA-256 mapping, so it carries that hash's 256-bit width.
const SALT_BYTES: usize = 32;
const HEX_DIGITS: [char; 16] = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
];

/// The operating system's random source failed.
#[derive(Debug, thiserror::Error)]
#[error("the operating system random source failed: {0}")]
pub struct SaltError(getrandom::Error);

impl AnonymizeOptions {
    /// Default options for `profile` with a UID salt drawn from the operating
    /// system.
    ///
    /// [`AnonymizeOptions::default`] carries a fixed, public salt: every export
    /// made with it maps one source UID to the same replacement, so two exports
    /// can be linked to each other and a known source UID can be confirmed
    /// against an export. A fresh salt removes both links. The salt is not
    /// stored anywhere, so the mapping cannot be reproduced after the run; keep
    /// [`AnonymizeOptions::uid_salt`] when a later export must match this one.
    ///
    /// # Errors
    /// Returns [`SaltError`] when the operating system supplies no randomness.
    pub fn with_fresh_salt(profile: AnonymizationProfile) -> Result<Self, SaltError> {
        let mut bytes = [0_u8; SALT_BYTES];
        getrandom::fill(&mut bytes).map_err(SaltError)?;
        let mut uid_salt = String::with_capacity(SALT_BYTES * 2);
        for byte in bytes {
            uid_salt.push(HEX_DIGITS[usize::from(byte >> 4)]);
            uid_salt.push(HEX_DIGITS[usize::from(byte & 0x0F)]);
        }
        Ok(Self {
            profile,
            uid_salt,
            ..Self::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{AnonymizationProfile, AnonymizeOptions, SALT_BYTES};

    #[test]
    fn fresh_salts_are_full_width_hex_and_differ_between_calls() -> Result<(), super::SaltError> {
        let first = AnonymizeOptions::with_fresh_salt(AnonymizationProfile::Enhanced)?;
        let second = AnonymizeOptions::with_fresh_salt(AnonymizationProfile::Enhanced)?;

        assert!(matches!(first.profile, AnonymizationProfile::Enhanced));
        assert_eq!(first.uid_salt.len(), SALT_BYTES * 2);
        assert!(first.uid_salt.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_ne!(first.uid_salt, second.uid_salt);
        assert_eq!(first.patient_id, AnonymizeOptions::default().patient_id);
        Ok(())
    }
}
