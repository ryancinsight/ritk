//! DICOM DIMSE Service Class User (SCU) networking primitives (GAP-262-IO-01).

pub mod association;
pub(crate) mod command;
pub mod context;
pub mod dimse;
mod echo;
mod find;
mod move_;
pub mod pdu;
pub mod scp;
mod store;
pub mod types;

#[cfg(test)]
#[path = "tests_dimse.rs"]
mod tests_dimse;

#[cfg(test)]
#[path = "tests_dimse_association.rs"]
mod tests_dimse_association;

pub use association::{Association, FindResult, MoveResult};
pub use context::{AssociationConfig, NegotiatedContext, RequestedPresentationContext};
pub use dimse::{CommandField, DimseMessage, DimseStatus};
pub use pdu::{AssociateAcPdu, AssociateRqPdu, Pdu};
pub use types::{
    AeTitle, DicomAddress, EchoResponse, MoveResponse, NetworkingError, StoreResponse,
};

pub use command::parse_dataset_ivr_le;
pub use echo::echo;
pub use find::{find, FindLevel, FindQuery};
pub use move_::{retrieve, retrieve_series, MoveDestination};
pub use scp::{ScpConfig, StoreScp, StoreScpHandle, StoredInstance};
pub use store::store;
