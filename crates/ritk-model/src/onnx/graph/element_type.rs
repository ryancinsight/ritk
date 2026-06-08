//! ONNX element data types.

/// ONNX element data types.
///
/// Reference: <https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L484>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxElementType {
    /// 32-bit float
    Float,
    /// Unsigned 8-bit integer
    Uint8,
    /// Signed 8-bit integer
    Int8,
    /// Unsigned 16-bit integer
    Uint16,
    /// Signed 16-bit integer
    Int16,
    /// Signed 32-bit integer
    Int32,
    /// Signed 64-bit integer
    Int64,
    /// String (not supported for tensors)
    String,
    /// Boolean
    Bool,
    /// 16-bit float (IEEE 754-2008 binary16)
    Float16,
    /// 64-bit float
    Float64,
    /// Unsigned 32-bit integer
    Uint32,
    /// Unsigned 64-bit integer
    Uint64,
    /// Brain floating point (bfloat16)
    Bfloat16,
    /// Complex with 32-bit float real and imaginary parts
    Complex64,
    /// Complex with 64-bit float real and imaginary parts
    Complex128,
}

impl OnnxElementType {
    /// Get the size of each element in bytes.
    pub fn element_size(&self) -> usize {
        match self {
            OnnxElementType::Float => 4,
            OnnxElementType::Uint8 => 1,
            OnnxElementType::Int8 => 1,
            OnnxElementType::Uint16 => 2,
            OnnxElementType::Int16 => 2,
            OnnxElementType::Int32 => 4,
            OnnxElementType::Int64 => 8,
            OnnxElementType::String => 0, // Variable size
            OnnxElementType::Bool => 1,
            OnnxElementType::Float16 => 2,
            OnnxElementType::Float64 => 8,
            OnnxElementType::Uint32 => 4,
            OnnxElementType::Uint64 => 8,
            OnnxElementType::Bfloat16 => 2,
            OnnxElementType::Complex64 => 8,
            OnnxElementType::Complex128 => 16,
        }
    }

    /// Check if this is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            OnnxElementType::Float
                | OnnxElementType::Float16
                | OnnxElementType::Float64
                | OnnxElementType::Bfloat16
        )
    }

    /// Check if this is an integer type.
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            OnnxElementType::Uint8
                | OnnxElementType::Int8
                | OnnxElementType::Uint16
                | OnnxElementType::Int16
                | OnnxElementType::Int32
                | OnnxElementType::Int64
                | OnnxElementType::Uint32
                | OnnxElementType::Uint64
        )
    }
}

impl std::fmt::Display for OnnxElementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxElementType::Float => write!(f, "float32"),
            OnnxElementType::Uint8 => write!(f, "uint8"),
            OnnxElementType::Int8 => write!(f, "int8"),
            OnnxElementType::Uint16 => write!(f, "uint16"),
            OnnxElementType::Int16 => write!(f, "int16"),
            OnnxElementType::Int32 => write!(f, "int32"),
            OnnxElementType::Int64 => write!(f, "int64"),
            OnnxElementType::String => write!(f, "string"),
            OnnxElementType::Bool => write!(f, "bool"),
            OnnxElementType::Float16 => write!(f, "float16"),
            OnnxElementType::Float64 => write!(f, "float64"),
            OnnxElementType::Uint32 => write!(f, "uint32"),
            OnnxElementType::Uint64 => write!(f, "uint64"),
            OnnxElementType::Bfloat16 => write!(f, "bfloat16"),
            OnnxElementType::Complex64 => write!(f, "complex64"),
            OnnxElementType::Complex128 => write!(f, "complex128"),
        }
    }
}
