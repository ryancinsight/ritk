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
    /// A Coeus backend operation, tensor constructor, or layer
    /// initialization failed.
    ///
    /// The boxed source is the backend-typed failure (`B::Error` or
    /// [`coeus_nn::init::InitializationError`]) with its chain intact. A `From`
    /// impl is impossible here: the backend error is an associated type, so
    /// a blanket impl would overlap [`ModelError`] conversion above; use
    /// [`ModelError::backend`] at fallible call sites instead.
    #[error(transparent)]
    Backend(Box<dyn std::error::Error + Send + Sync>),
}

impl ModelError {
    /// Box a backend-typed failure into a [`ModelError::Backend`].
    pub fn backend<E>(error: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Backend(Box::new(error))
    }
}

impl<E> From<coeus_nn::ModuleError<E>> for ModelError
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn from(error: coeus_nn::ModuleError<E>) -> Self {
        Self::Module(Box::new(error))
    }
}
