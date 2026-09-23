//! The elements of the GIFTI DTD (section 8.1.2) and where each may appear.

/// Deepest element nesting accepted. The DTD's deepest path is five
/// (`GIFTI/DataArray/MetaData/MD/Name`); unknown extension elements get room
/// to nest below that, but not without bound.
pub(super) const MAX_DEPTH: usize = 32;

/// The elements of the DTD, plus everything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Element {
    Gifti,
    MetaData,
    Md,
    Name,
    Value,
    LabelTable,
    Label,
    DataArray,
    Transform,
    DataSpace,
    TransformedSpace,
    MatrixData,
    Data,
    Unknown,
}

impl Element {
    pub(super) fn from_name(name: &[u8]) -> Self {
        match name {
            b"GIFTI" => Self::Gifti,
            b"MetaData" => Self::MetaData,
            b"MD" => Self::Md,
            b"Name" => Self::Name,
            b"Value" => Self::Value,
            b"LabelTable" => Self::LabelTable,
            b"Label" => Self::Label,
            b"DataArray" => Self::DataArray,
            b"CoordinateSystemTransformMatrix" => Self::Transform,
            b"DataSpace" => Self::DataSpace,
            b"TransformedSpace" => Self::TransformedSpace,
            b"MatrixData" => Self::MatrixData,
            b"Data" => Self::Data,
            _ => Self::Unknown,
        }
    }

    pub(super) const fn tag(self) -> &'static str {
        match self {
            Self::Gifti => "GIFTI",
            Self::MetaData => "MetaData",
            Self::Md => "MD",
            Self::Name => "Name",
            Self::Value => "Value",
            Self::LabelTable => "LabelTable",
            Self::Label => "Label",
            Self::DataArray => "DataArray",
            Self::Transform => "CoordinateSystemTransformMatrix",
            Self::DataSpace => "DataSpace",
            Self::TransformedSpace => "TransformedSpace",
            Self::MatrixData => "MatrixData",
            Self::Data => "Data",
            Self::Unknown => "unknown element",
        }
    }

    /// Whether `parent` may contain this element, per the DTD.
    pub(super) fn allowed_in(self, parent: Option<Self>) -> bool {
        match self {
            Self::Gifti => parent.is_none(),
            Self::MetaData => matches!(parent, Some(Self::Gifti | Self::DataArray)),
            Self::Md => parent == Some(Self::MetaData),
            Self::Name | Self::Value => parent == Some(Self::Md),
            Self::LabelTable | Self::DataArray => parent == Some(Self::Gifti),
            Self::Label => parent == Some(Self::LabelTable),
            Self::Transform | Self::Data => parent == Some(Self::DataArray),
            Self::DataSpace | Self::TransformedSpace | Self::MatrixData => {
                parent == Some(Self::Transform)
            }
            Self::Unknown => parent.is_some(),
        }
    }

    /// Whether the element's character content is data this reader keeps.
    pub(super) const fn captures_text(self) -> bool {
        matches!(
            self,
            Self::Name
                | Self::Value
                | Self::Label
                | Self::DataSpace
                | Self::TransformedSpace
                | Self::MatrixData
                | Self::Data
        )
    }
}
