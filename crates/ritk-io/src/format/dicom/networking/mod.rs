//! DICOM DIMSE Service Class User (SCU) networking primitives (GAP-262-IO-01).
//!
//! Implements the four core DIMSE service operations over DICOM Upper Layer
//! (UL) TCP associations against a standard PACS:
//!
//! | Service | SOP Class | Function |
//! |---------|-----------|----------|
//! | C-ECHO | 1.2.840.10008.1.1 | Verify PACS connectivity |
//! | C-FIND | 1.2.840.10008.5.1.4.1.2.2.1 | Query studies/series/instances |
//! | C-STORE | Storage SOP class of object | Transmit DICOM object to PACS |
//! | C-MOVE | 1.2.840.10008.5.1.4.1.2.2.2 | Request PACS-to-AE transfer |
//!
//! # Usage
//! ```ignore
//! use ritk_io::format::dicom::networking::{
//!     AeTitle, AssociationConfig, DicomAddress, echo,
//! };
//!
//! let config = AssociationConfig::new(
//!     AeTitle::new("MY_SCU").unwrap(),
//!     DicomAddress::new("pacs.hospital.org", 104, AeTitle::new("PACS_AET").unwrap()),
//! );
//! let resp = echo(&config)?;
//! assert_eq!(resp.status, 0x0000);
//! ```
//!
//! # Reference
//! - PS3.4 Service Class Specifications
//! - PS3.7 Message Exchange (DIMSE protocol)
//! - PS3.8 Network Communication Support for Message Exchange

pub mod association;
pub(crate) mod command;
mod echo;
mod find;
mod move_;
mod store;

#[cfg(test)]
mod tests_dimse;

pub use association::{
    AeTitle, AssociationConfig, DicomAddress, EchoResponse, FindResult, MoveResponse,
    NetworkingError, StoreResponse,
};
pub use echo::echo;
pub use find::{find, FindLevel, FindQuery};
pub use move_::{retrieve, MoveDestination};
pub use store::store;
