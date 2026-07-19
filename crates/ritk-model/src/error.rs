//! Model graph contract failures.

/// Failure produced while evaluating a registration model graph.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// A tensor does not satisfy an operation's rank or axis contract.
    #[error("{operation} expected {expected}, got shape {actual:?}")]
    Shape {
        /// Operation whose input contract failed.
        operation: &'static str,
        /// Expected shape contract.
        expected: &'static str,
        /// Actual runtime shape.
        actual: Vec<usize>,
    },
    /// Coordinate-grid interpolation rejected its inputs.
    #[error(transparent)]
    Interpolation(#[from] coeus_ops::InterpolationError),
}
