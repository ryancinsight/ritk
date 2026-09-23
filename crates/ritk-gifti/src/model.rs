//! The GIFTI document model (GIFTI 1.0, section 2).

use crate::GiftiError;

/// Most dimensions a data array may declare: the DTD defines `Dim0` to `Dim5`
/// (section 8.1.2).
pub(crate) const MAX_DIMENSIONS: usize = 6;

/// Most elements a data array may hold.
///
/// A left-hemisphere time series in the specification's own examples holds
/// 143,479 × 136 ≈ 2·10⁷ values (section 14.3); five times that admits any
/// real surface file while keeping a forged shape from demanding more than a
/// few hundred megabytes before its payload is checked.
pub(crate) const MAX_ELEMENTS: usize = 100_000_000;

/// Ordered `Name`/`Value` metadata pairs (section 2.9–2.10).
///
/// Order and unrecognised names are kept, because the specification asks
/// that metadata an application does not understand be passed through.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetaData {
    entries: Vec<(String, String)>,
}

impl MetaData {
    /// Metadata from `(name, value)` pairs, in order.
    #[must_use]
    pub fn new(entries: impl IntoIterator<Item = (String, String)>) -> Self {
        Self {
            entries: entries.into_iter().collect(),
        }
    }

    /// The value of the first entry called `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    /// Every entry, in document order.
    #[must_use]
    pub fn entries(&self) -> &[(String, String)] {
        &self.entries
    }

    pub(crate) fn push(&mut self, name: String, value: String) {
        self.entries.push((name, value));
    }
}

/// What a data array holds (section 2.3.4.9).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Intent {
    /// `NIFTI_INTENT_NONE`: unspecified.
    None,
    /// `NIFTI_INTENT_POINTSET`: vertex coordinates, shape `[n, 3]`.
    PointSet,
    /// `NIFTI_INTENT_TRIANGLE`: vertex-index triplets, shape `[m, 3]`.
    Triangle,
    /// `NIFTI_INTENT_LABEL`: keys into the label table, one per vertex.
    Label,
    /// `NIFTI_INTENT_NODE_INDEX`: the vertices a sparse file covers.
    NodeIndex,
    /// `NIFTI_INTENT_SHAPE`: shape measurements such as curvature.
    Shape,
    /// `NIFTI_INTENT_TIME_SERIES`: one value per vertex per time point.
    TimeSeries,
    /// `NIFTI_INTENT_VECTOR`: three-dimensional vectors.
    Vector,
    /// Any other intent, statistical ones included, by its name.
    Other(String),
}

impl Intent {
    /// The intent named `name` in a document.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        match name {
            "NIFTI_INTENT_NONE" => Self::None,
            "NIFTI_INTENT_POINTSET" => Self::PointSet,
            "NIFTI_INTENT_TRIANGLE" => Self::Triangle,
            "NIFTI_INTENT_LABEL" => Self::Label,
            "NIFTI_INTENT_NODE_INDEX" => Self::NodeIndex,
            "NIFTI_INTENT_SHAPE" => Self::Shape,
            "NIFTI_INTENT_TIME_SERIES" => Self::TimeSeries,
            "NIFTI_INTENT_VECTOR" => Self::Vector,
            other => Self::Other(other.to_owned()),
        }
    }

    /// The name the document uses.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::None => "NIFTI_INTENT_NONE",
            Self::PointSet => "NIFTI_INTENT_POINTSET",
            Self::Triangle => "NIFTI_INTENT_TRIANGLE",
            Self::Label => "NIFTI_INTENT_LABEL",
            Self::NodeIndex => "NIFTI_INTENT_NODE_INDEX",
            Self::Shape => "NIFTI_INTENT_SHAPE",
            Self::TimeSeries => "NIFTI_INTENT_TIME_SERIES",
            Self::Vector => "NIFTI_INTENT_VECTOR",
            Self::Other(name) => name,
        }
    }
}

/// Element order of a multi-dimensional array (section 2.3.4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexingOrder {
    /// The last index varies fastest (C order).
    RowMajor,
    /// The first index varies fastest (Fortran order).
    ColumnMajor,
}

/// How [`GiftiImage::write`] encodes each `Data` element (section 2.3.4.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataEncoding {
    /// Whitespace-separated decimal text.
    Ascii,
    /// Base64 over little-endian binary.
    Base64Binary,
    /// Base64 over a zlib stream of little-endian binary (section 5.0).
    GZipBase64Binary,
}

/// The values of a data array, in one of the three GIFTI types
/// (section 2.3.4.2).
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayData {
    /// `NIFTI_TYPE_UINT8`.
    UInt8(Box<[u8]>),
    /// `NIFTI_TYPE_INT32`.
    Int32(Box<[i32]>),
    /// `NIFTI_TYPE_FLOAT32`.
    Float32(Box<[f32]>),
}

impl ArrayData {
    /// Number of values.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::UInt8(values) => values.len(),
            Self::Int32(values) => values.len(),
            Self::Float32(values) => values.len(),
        }
    }

    /// Whether there are no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `DataType` attribute naming this type.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::UInt8(_) => "NIFTI_TYPE_UINT8",
            Self::Int32(_) => "NIFTI_TYPE_INT32",
            Self::Float32(_) => "NIFTI_TYPE_FLOAT32",
        }
    }

    /// The same values permuted by `source_of`, where `source_of(i)` is the
    /// position in `self` of output element `i`.
    fn permuted(&self, source_of: impl Fn(usize) -> usize) -> Self {
        fn gather<T: Copy>(values: &[T], source_of: impl Fn(usize) -> usize) -> Box<[T]> {
            (0..values.len())
                .map(|index| values[source_of(index)])
                .collect()
        }
        match self {
            Self::UInt8(values) => Self::UInt8(gather(values, source_of)),
            Self::Int32(values) => Self::Int32(gather(values, source_of)),
            Self::Float32(values) => Self::Float32(gather(values, source_of)),
        }
    }
}

/// A transform from the data's stereotaxic space to another
/// (section 2.1, 2.4, 2.8, 2.12).
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateTransform {
    /// Space of the stored coordinates, e.g. `NIFTI_XFORM_TALAIRACH`.
    pub data_space: String,
    /// Space the matrix maps them into.
    pub transformed_space: String,
    /// The 4×4 matrix, rows first.
    pub matrix: [[f64; 4]; 4],
}

/// One data array: an intent, a shape, and values (section 2.3).
#[derive(Debug, Clone, PartialEq)]
pub struct DataArray {
    intent: Intent,
    dims: Box<[usize]>,
    order: IndexingOrder,
    data: ArrayData,
    metadata: MetaData,
    transforms: Vec<CoordinateTransform>,
}

impl DataArray {
    /// A row-major array of shape `dims`.
    ///
    /// # Errors
    ///
    /// [`GiftiError::Structure`] when `dims` has no entries or more than six,
    /// or its product is not the number of values.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_gifti::{ArrayData, DataArray, Intent};
    ///
    /// let thickness = DataArray::new(Intent::Shape, vec![2], ArrayData::Float32(vec![2.5, 3.0].into()))?;
    /// assert_eq!(thickness.dims(), &[2]);
    /// # Ok::<(), ritk_gifti::GiftiError>(())
    /// ```
    pub fn new(intent: Intent, dims: Vec<usize>, data: ArrayData) -> Result<Self, GiftiError> {
        Self::with_order(intent, dims, IndexingOrder::RowMajor, data)
    }

    /// An array whose values are stored in `order`.
    ///
    /// # Errors
    ///
    /// As [`DataArray::new`].
    pub fn with_order(
        intent: Intent,
        dims: Vec<usize>,
        order: IndexingOrder,
        data: ArrayData,
    ) -> Result<Self, GiftiError> {
        let expected = element_count(&dims)?;
        if expected != data.len() {
            return Err(GiftiError::structure(
                "DataArray",
                format!(
                    "shape {dims:?} holds {expected} values, data has {}",
                    data.len()
                ),
            ));
        }
        Ok(Self {
            intent,
            dims: dims.into_boxed_slice(),
            order,
            data,
            metadata: MetaData::default(),
            transforms: Vec::new(),
        })
    }

    /// The same array carrying `metadata`.
    #[must_use]
    pub fn with_metadata(mut self, metadata: MetaData) -> Self {
        self.metadata = metadata;
        self
    }

    /// The same array carrying `transforms`.
    #[must_use]
    pub fn with_transforms(mut self, transforms: Vec<CoordinateTransform>) -> Self {
        self.transforms = transforms;
        self
    }

    /// What the array holds.
    #[must_use]
    pub const fn intent(&self) -> &Intent {
        &self.intent
    }

    /// The shape, `Dim0` first.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// The order the values are stored in.
    #[must_use]
    pub const fn order(&self) -> IndexingOrder {
        self.order
    }

    /// The values, in [`DataArray::order`].
    #[must_use]
    pub const fn data(&self) -> &ArrayData {
        &self.data
    }

    /// The values in row-major order, transposing a column-major array.
    #[must_use]
    pub fn row_major(&self) -> ArrayData {
        match self.order {
            IndexingOrder::RowMajor => self.data.clone(),
            IndexingOrder::ColumnMajor => {
                let dims = &self.dims;
                let rank = dims.len();
                // Row-major strides: the last axis is contiguous.
                let mut row_strides = vec![1_usize; rank];
                for axis in (1..rank).rev() {
                    row_strides[axis - 1] = row_strides[axis] * dims[axis];
                }
                // Column-major strides: the first axis is contiguous.
                let mut column_strides = vec![1_usize; rank];
                for axis in 1..rank {
                    column_strides[axis] = column_strides[axis - 1] * dims[axis - 1];
                }
                self.data.permuted(|row_major_index| {
                    (0..rank)
                        .map(|axis| {
                            row_major_index / row_strides[axis] % dims[axis] * column_strides[axis]
                        })
                        .sum()
                })
            }
        }
    }

    /// Array metadata.
    #[must_use]
    pub const fn metadata(&self) -> &MetaData {
        &self.metadata
    }

    /// Coordinate transforms, meaningful for [`Intent::PointSet`].
    #[must_use]
    pub fn transforms(&self) -> &[CoordinateTransform] {
        &self.transforms
    }
}

/// The product of `dims`, bounded by [`MAX_ELEMENTS`].
pub(crate) fn element_count(dims: &[usize]) -> Result<usize, GiftiError> {
    if dims.is_empty() || dims.len() > MAX_DIMENSIONS {
        return Err(GiftiError::structure(
            "DataArray",
            format!("{} dimensions, expected 1..={MAX_DIMENSIONS}", dims.len()),
        ));
    }
    dims.iter()
        .try_fold(1_usize, |product, extent| product.checked_mul(*extent))
        .filter(|count| *count <= MAX_ELEMENTS)
        .ok_or_else(|| {
            GiftiError::structure(
                "DataArray",
                format!("shape {dims:?} exceeds {MAX_ELEMENTS} values"),
            )
        })
}

/// One entry of the label table (section 2.6).
#[derive(Debug, Clone, PartialEq)]
pub struct GiftiLabel {
    key: u32,
    name: String,
    rgba: Option<[f32; 4]>,
}

impl GiftiLabel {
    /// A label; `rgba` components lie in `0.0..=1.0`.
    ///
    /// # Errors
    ///
    /// [`GiftiError::Structure`] for a colour component outside `0.0..=1.0`.
    pub fn new(key: u32, name: String, rgba: Option<[f32; 4]>) -> Result<Self, GiftiError> {
        if let Some(component) = rgba
            .iter()
            .flatten()
            .find(|component| !(0.0..=1.0).contains(*component))
        {
            return Err(GiftiError::structure(
                "Label",
                format!("key {key}: colour component {component} outside 0..=1"),
            ));
        }
        Ok(Self { key, name, rgba })
    }

    /// The key data arrays of intent [`Intent::Label`] refer to.
    #[must_use]
    pub const fn key(&self) -> u32 {
        self.key
    }

    /// The label name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Red, green, blue, alpha in `0.0..=1.0`, when the file gives a colour.
    #[must_use]
    pub const fn rgba(&self) -> Option<[f32; 4]> {
        self.rgba
    }
}

/// A GIFTI document (section 2.5).
#[derive(Debug, Clone, PartialEq)]
pub struct GiftiImage {
    pub(crate) metadata: MetaData,
    pub(crate) labels: Vec<GiftiLabel>,
    pub(crate) arrays: Vec<DataArray>,
}

impl GiftiImage {
    /// A document from its parts.
    ///
    /// # Errors
    ///
    /// [`GiftiError::Structure`] when two labels share a key.
    pub fn new(
        metadata: MetaData,
        labels: Vec<GiftiLabel>,
        arrays: Vec<DataArray>,
    ) -> Result<Self, GiftiError> {
        let mut keys: Vec<u32> = labels.iter().map(GiftiLabel::key).collect();
        keys.sort_unstable();
        if let Some(pair) = keys.windows(2).find(|pair| pair.first() == pair.last()) {
            return Err(GiftiError::structure(
                "LabelTable",
                format!("key {:?} appears twice", pair.first()),
            ));
        }
        Ok(Self {
            metadata,
            labels,
            arrays,
        })
    }

    /// File metadata.
    #[must_use]
    pub const fn metadata(&self) -> &MetaData {
        &self.metadata
    }

    /// The label table, in document order.
    #[must_use]
    pub fn labels(&self) -> &[GiftiLabel] {
        &self.labels
    }

    /// The data arrays, in document order.
    #[must_use]
    pub fn arrays(&self) -> &[DataArray] {
        &self.arrays
    }
}
