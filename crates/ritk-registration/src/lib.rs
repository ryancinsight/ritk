pub mod metric;
pub mod optimizer;
pub mod registration;
pub mod multires;
pub mod regularization;
// pub mod multiresolution;
pub mod error;
pub mod validation;
pub mod progress;
// pub mod enhanced_registration;

pub use error::{RegistrationError, Result};
pub use validation::{ValidationConfig, ConvergenceChecker};
pub use progress::{ProgressCallback, ProgressTracker, ConsoleProgressCallback, HistoryCallback, EarlyStoppingCallback, ProgressInfo};
// pub use enhanced_registration::{EnhancedRegistration, RegistrationConfig};
