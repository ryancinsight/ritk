//! The error every GIFTI operation returns.

/// Error returned when reading or writing a GIFTI document.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GiftiError {
    /// The underlying reader or writer failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The document is not well-formed XML.
    #[error("XML error at byte {position}: {reason}")]
    Xml {
        /// Byte offset the parser had reached.
        position: u64,
        /// The parser's description.
        reason: String,
    },

    /// The document is well-formed but breaks the GIFTI structure: a missing
    /// or invalid attribute, a misplaced element, or inconsistent counts.
    #[error("{element}: {reason}")]
    Structure {
        /// The element at fault.
        element: &'static str,
        /// What is wrong with it.
        reason: String,
    },

    /// A `Data` payload does not decode to what its `DataArray` declares.
    #[error("data array {array}: {reason}")]
    Data {
        /// Zero-based position of the data array in the document.
        array: usize,
        /// What is wrong with the payload.
        reason: String,
    },

    /// A feature of the format this implementation does not provide.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl GiftiError {
    /// A [`GiftiError::Structure`] with `reason` rendered.
    pub(crate) fn structure(element: &'static str, reason: impl Into<String>) -> Self {
        Self::Structure {
            element,
            reason: reason.into(),
        }
    }

    /// A [`GiftiError::Data`] with `reason` rendered.
    pub(crate) fn data(array: usize, reason: impl Into<String>) -> Self {
        Self::Data {
            array,
            reason: reason.into(),
        }
    }
}
