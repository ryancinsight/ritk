pub mod metric;
pub mod multires;
pub mod optimizer;
pub mod registration;
pub mod regularization;
// pub mod multiresolution;
pub mod error;
pub mod progress;
pub mod validation;
// pub mod enhanced_registration;

pub use error::{RegistrationError, Result};
pub use progress::{
    ConsoleProgressCallback, EarlyStoppingCallback, HistoryCallback, ProgressCallback,
    ProgressInfo, ProgressTracker,
};
pub use progress::ConvergenceChecker;
pub use validation::ValidationConfig;
// pub use enhanced_registration::{EnhancedRegistration, RegistrationConfig};
