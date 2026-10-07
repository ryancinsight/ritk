//! Resource ceilings for DICOM discovery and decoding.

use anyhow::{bail, Result};
use ritk_dicom::ParseBudget;

const DEFAULT_MAX_RETAINED_BYTES: usize = 1024 * 1024 * 1024;
const DEFAULT_MAX_DECODED_BYTES: usize = 1024 * 1024 * 1024;

/// Resource ceilings for one DICOM read workflow.
///
/// The parser budget limits each encoded input and its structural traversal.
/// The retained ceiling bounds encoded study bytes. The decoded ceiling bounds
/// the reader's planned sample, frame, geometry, and calibration workspace;
/// it is not a process-RSS measurement. Keeping these ceilings separate lets a
/// caller admit a large study made of small instances without allowing one
/// malformed instance to request the study's whole storage allowance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DicomReadBudget {
    parser: ParseBudget,
    max_retained_bytes: usize,
    max_decoded_bytes: usize,
    max_instances: usize,
}

impl Default for DicomReadBudget {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl DicomReadBudget {
    /// Default ceilings for a desktop or browser-backed viewer workflow.
    ///
    /// The one-gibibyte retained and decoded ceilings are explicit finite
    /// bounds; callers handling constrained devices should construct a
    /// smaller budget with [`Self::try_new`].
    pub const DEFAULT: Self = Self {
        parser: ParseBudget::DEFAULT,
        max_retained_bytes: DEFAULT_MAX_RETAINED_BYTES,
        max_decoded_bytes: DEFAULT_MAX_DECODED_BYTES,
        max_instances: ParseBudget::DEFAULT.max_elements,
    };

    /// Construct a workflow budget with explicit storage ceilings.
    ///
    /// # Errors
    ///
    /// Returns an error when either workflow ceiling is zero.
    pub fn try_new(
        parser: ParseBudget,
        max_retained_bytes: usize,
        max_decoded_bytes: usize,
    ) -> Result<Self> {
        Self::try_new_with_max_instances(
            parser,
            max_retained_bytes,
            max_decoded_bytes,
            parser.max_elements,
        )
    }

    /// Construct a workflow budget with an explicit candidate-instance ceiling.
    ///
    /// # Errors
    ///
    /// Returns an error when any workflow ceiling is zero.
    pub fn try_new_with_max_instances(
        parser: ParseBudget,
        max_retained_bytes: usize,
        max_decoded_bytes: usize,
        max_instances: usize,
    ) -> Result<Self> {
        if max_retained_bytes == 0 {
            bail!("DICOM retained-byte budget must be nonzero");
        }
        if max_decoded_bytes == 0 {
            bail!("DICOM decoded-byte budget must be nonzero");
        }
        if max_instances == 0 {
            bail!("DICOM instance-count budget must be nonzero");
        }
        Ok(Self {
            parser,
            max_retained_bytes,
            max_decoded_bytes,
            max_instances,
        })
    }

    /// Return the structural parser budget used for each instance.
    #[must_use]
    pub const fn parser(&self) -> ParseBudget {
        self.parser
    }

    /// Return the maximum retained encoded bytes for one study.
    #[must_use]
    pub const fn max_retained_bytes(&self) -> usize {
        self.max_retained_bytes
    }

    /// Return the maximum decoded workspace bytes for one load.
    #[must_use]
    pub const fn max_decoded_bytes(&self) -> usize {
        self.max_decoded_bytes
    }

    /// Return the maximum candidate instances admitted by one discovery.
    #[must_use]
    pub const fn max_instances(&self) -> usize {
        self.max_instances
    }

    /// Check a cumulative retained-byte total before storing another member.
    ///
    /// # Errors
    ///
    /// Returns an error when `bytes` exceeds the study ceiling.
    pub fn checked_retained_bytes(&self, bytes: usize) -> Result<usize> {
        if bytes > self.max_retained_bytes {
            bail!(
                "DICOM retained study bytes {bytes} exceed budget {}",
                self.max_retained_bytes
            );
        }
        Ok(bytes)
    }

    /// Check decoded workspace bytes before allocating a volume or frame set.
    ///
    /// # Errors
    ///
    /// Returns an error when `bytes` exceeds the decoded workspace ceiling.
    pub fn checked_decoded_bytes(&self, bytes: usize) -> Result<usize> {
        if bytes > self.max_decoded_bytes {
            bail!(
                "DICOM decoded workspace bytes {bytes} exceed budget {}",
                self.max_decoded_bytes
            );
        }
        Ok(bytes)
    }

    /// Check a candidate-instance count before parsing any catalog member.
    ///
    /// # Errors
    ///
    /// Returns an error when `instances` exceeds the workflow ceiling.
    pub fn checked_instances(&self, instances: usize) -> Result<usize> {
        if instances > self.max_instances {
            bail!(
                "DICOM candidate instance count {instances} exceeds budget {}",
                self.max_instances
            );
        }
        Ok(instances)
    }
}
