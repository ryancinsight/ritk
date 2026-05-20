//! MONAI Label Server REST client.
//!
//! Provides synchronous HTTP access to the MONAI Label Server inference API
//! (<https://docs.monai.io/projects/label/en/latest/apis.html>):
//!
//! - [`MonaiLabelClient::info`] — server metadata
//! - [`MonaiLabelClient::models`] — available segmentation models
//! - [`MonaiLabelClient::infer`] — run AI inference on a stored image
//!
//! # Example
//!
//! ```rust,ignore
//! use ritk_model::monai::{MonaiLabelClient, InferRequest};
//!
//! let client = MonaiLabelClient::new("http://localhost:8000");
//! let info = client.info()?;
//! println!("Server: {} v{}", info.name, info.version);
//! let models = client.models()?;
//! let req = InferRequest::new("segmentation_ct_lungs", "patient-001-ct");
//! let resp = client.infer(&req)?;
//! // resp.label contains the NIfTI segmentation mask bytes
//! ```

pub mod client;
pub mod multipart;
pub mod types;

pub use client::MonaiLabelClient;
pub use types::{InferRequest, InferResponse, ModelInfo, ModelType, MonaiError, ServerInfo};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
