//! Browser event provenance policy for the viewer presentation boundary.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BrowserEventTrust {
    Trusted,
    Untrusted,
}

#[cfg(target_arch = "wasm32")]
impl From<metis_web::CanvasEventTrust> for BrowserEventTrust {
    fn from(value: metis_web::CanvasEventTrust) -> Self {
        if value.is_trusted() {
            Self::Trusted
        } else {
            Self::Untrusted
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BrowserEventDisposition {
    Admit,
    Drop,
}

pub(crate) const fn browser_event_disposition(trust: BrowserEventTrust) -> BrowserEventDisposition {
    if matches!(trust, BrowserEventTrust::Trusted) {
        BrowserEventDisposition::Admit
    } else {
        BrowserEventDisposition::Drop
    }
}

#[cfg(test)]
mod tests {
    use super::{browser_event_disposition, BrowserEventDisposition, BrowserEventTrust};

    #[test]
    fn browser_provenance_admits_only_trusted_events() {
        assert_eq!(
            browser_event_disposition(BrowserEventTrust::Trusted),
            BrowserEventDisposition::Admit
        );
        assert_eq!(
            browser_event_disposition(BrowserEventTrust::Untrusted),
            BrowserEventDisposition::Drop
        );
    }
}
