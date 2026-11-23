//! Scope and metadata profile shared by desktop and browser exports.
use ritk_io::format::dicom::anonymize::AnonymizationProfile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AnonymizationScope {
    Study,
    Series,
}

/// Which RITK anonymization profile a de-identified export applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::presentation::native_session) enum Deidentification {
    /// [`AnonymizationProfile::BasicReplaceUids`]: study, series, acquisition
    /// and content dates and times and the study and series descriptions stay.
    Basic,
    /// [`AnonymizationProfile::Enhanced`]: those dates and times are emptied
    /// and the study and series descriptions are removed.
    Strict,
}

impl Deidentification {
    pub(super) const fn profile(self) -> AnonymizationProfile {
        match self {
            Self::Basic => AnonymizationProfile::BasicReplaceUids,
            Self::Strict => AnonymizationProfile::Enhanced,
        }
    }

    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Basic => "Basic",
            Self::Strict => "Strict",
        }
    }
}
