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
    /// A Coeus module rejected its input or failed a backend operation.
    ///
    /// The boxed source is the typed [`coeus_nn::ModuleError`] with its
    /// chain intact; boxing erases only the backend type parameter so this
    /// enum stays non-generic across every model consumer.
    #[error(transparent)]
    Module(Box<dyn std::error::Error + Send + Sync>),
}

impl<E> From<coeus_nn::ModuleError<E>> for ModelError
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn from(error: coeus_nn::ModuleError<E>) -> Self {
        Self::Module(Box::new(error))
    }
}
