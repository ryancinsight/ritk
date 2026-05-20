//! MONAI Label Server domain types.
//!
//! # Domain model
//!
//! | Type | Source |
//! |---|---|
//! | `ServerInfo` | GET /info response |
//! | `ModelInfo` | one entry from GET /models response |
//! | `ModelType` | discriminated string enum ("segmentation", "deepedit", etc.) |
//! | `InferRequest` | input to POST /infer/{model} |
//! | `InferResponse` | output from POST /infer/{model} |
//! | `MonaiError` | all failure modes |

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── ServerInfo ────────────────────────────────────────────────────────────────

/// MONAI Label Server metadata returned by `GET /info`.
///
/// Unknown fields in the JSON response are silently ignored by serde.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerInfo {
    /// Human-readable server name.
    pub name: String,
    /// Server description.
    #[serde(default)]
    pub description: String,
    /// Server version string (semver or custom).
    #[serde(default)]
    pub version: String,
    /// Label map or metadata; structure is model-specific — stored as raw JSON.
    #[serde(default)]
    pub labels: serde_json::Value,
}

// ── ModelType ─────────────────────────────────────────────────────────────────

/// Classification of a MONAI Label model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelType {
    /// Standard volumetric segmentation (e.g., organ or lesion).
    Segmentation,
    /// DeepEdit — interactive, annotation-driven segmentation.
    DeepEdit,
    /// Active learning query strategy.
    ActiveLearning,
    /// Any type string not in the above set.
    Unknown(String),
}

impl Default for ModelType {
    fn default() -> Self {
        Self::Unknown(String::new())
    }
}

impl<'de> Deserialize<'de> for ModelType {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(match s.as_str() {
            "segmentation" | "Segmentation" => Self::Segmentation,
            "deepedit" | "DeepEdit" => Self::DeepEdit,
            "activelearning" | "ActiveLearning" => Self::ActiveLearning,
            other => Self::Unknown(other.to_owned()),
        })
    }
}

impl Serialize for ModelType {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(match self {
            Self::Segmentation => "segmentation",
            Self::DeepEdit => "deepedit",
            Self::ActiveLearning => "activelearning",
            Self::Unknown(o) => o.as_str(),
        })
    }
}

// ── ModelInfo ─────────────────────────────────────────────────────────────────

/// Metadata for a single MONAI Label model, as returned in `GET /models`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier (injected from the JSON map key if absent in the value).
    pub name: String,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// Model category.
    #[serde(rename = "type", default)]
    pub model_type: ModelType,
    /// Output class labels in index order (index 0 = background by convention).
    #[serde(default)]
    pub labels: Vec<String>,
    /// Spatial dimensionality of the model input (2 or 3).
    #[serde(default)]
    pub dimension: u32,
}

// ── InferRequest ─────────────────────────────────────────────────────────────

/// Parameters for a single `POST /infer/{model}?image={image_id}` call.
#[derive(Debug, Clone)]
pub struct InferRequest {
    /// Name of the model to invoke (must match a key in `GET /models`).
    pub model: String,
    /// Identifier of the image in the server's datastore.
    pub image_id: String,
    /// Optional model-specific inference parameters; sent as JSON body.
    /// Use `serde_json::json!({})` for an empty parameter set.
    pub params: serde_json::Value,
}

impl InferRequest {
    /// Construct a request with an empty JSON params object.
    pub fn new(model: impl Into<String>, image_id: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            image_id: image_id.into(),
            params: serde_json::Value::Object(Default::default()),
        }
    }
}

// ── InferResponse ─────────────────────────────────────────────────────────────

/// Response from a `POST /infer/{model}` call.
///
/// MONAI Label Server returns multipart/form-data with two named parts:
/// - `label`  — binary NIfTI segmentation mask (typically `.nii.gz`).
/// - `params` — JSON metadata: timing, class probabilities, model version.
#[derive(Debug, Clone)]
pub struct InferResponse {
    /// Raw segmentation label bytes (NIfTI or NIfTI-compressed format).
    pub label: Vec<u8>,
    /// Inference metadata from the server (timing, scores, label map, etc.).
    pub params: serde_json::Value,
}

// ── MonaiError ────────────────────────────────────────────────────────────────

/// All failure modes for MONAI Label Server client operations.
#[derive(Debug, Error)]
pub enum MonaiError {
    /// Network or TLS transport failure from the underlying HTTP client.
    #[error("HTTP transport error: {0}")]
    Transport(#[from] reqwest::Error),

    /// JSON serialization or deserialization failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Server returned a non-2xx HTTP status code.
    #[error("server returned HTTP {status}: {body}")]
    ServerError {
        /// HTTP status code (e.g. 404, 500).
        status: u16,
        /// Response body text for diagnostic context.
        body: String,
    },

    /// Multipart or JSON response structure was unexpected.
    #[error("response parse error: {message}")]
    ParseError {
        /// Human-readable description of the parsing failure.
        message: String,
    },
}
