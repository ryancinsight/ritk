/// Attribute array attached to points or cells in a VTK dataset.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeArray {
    /// Scalar field. `values.len() == n_elements * num_components`.
    Scalars {
        values: Vec<f32>,
        num_components: usize },
    /// 3-component vector field. `values.len() == n_elements`.
    Vectors { values: Vec<[f32; 3]> },
    /// Unit-normal field. `values.len() == n_elements`.
    Normals { values: Vec<[f32; 3]> },
    /// Texture coordinate field. `values.len() == n_elements * dim`, `dim` in {1, 2, 3}.
    TextureCoords { values: Vec<f32>, dim: usize } }
