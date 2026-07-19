//! Spatial metadata transforms between MGH RAS header fields and RITK images.

use ritk_spatial::{Direction, Point, Spacing, Vector};

/// Whether the RAS (Right-Anterior-Superior) spatial metadata in the MGH
/// header is valid and should be used to derive image geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RasValidity {
    /// RAS fields are valid â€” use them to compute origin, spacing, direction.
    Valid,
    /// RAS fields are absent or unreliable â€” fall back to identity geometry.
    Synthetic,
}

pub(crate) fn derive_image_geometry(
    ras_validity: RasValidity,
    dims: [usize; 3],
    spacing_xyz: [f32; 3],
    direction_columns: [[f32; 3]; 3],
    c_ras: [f32; 3],
) -> (Spacing<3>, Direction<3>, Point<3>) {
    if ras_validity == RasValidity::Synthetic {
        return (
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
            Point::new([0.0, 0.0, 0.0]),
        );
    }

    let spacing = Spacing::new([
        spacing_xyz[0] as f64,
        spacing_xyz[1] as f64,
        spacing_xyz[2] as f64,
    ]);
    let direction = direction_matrix_from_columns(direction_columns);
    let origin_vec = Vector::new([c_ras[0] as f64, c_ras[1] as f64, c_ras[2] as f64])
        - centered_half_offset(direction, spacing, dims);

    (
        spacing,
        direction,
        Point::new([origin_vec[0], origin_vec[1], origin_vec[2]]),
    )
}

pub(crate) fn ras_center_from_geometry(
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
    shape_zyx: [usize; 3],
) -> Vector<3> {
    let dims_xyz = [shape_zyx[2], shape_zyx[1], shape_zyx[0]];
    Vector::new([origin[0], origin[1], origin[2]])
        + centered_half_offset(direction, spacing, dims_xyz)
}

fn direction_matrix_from_columns(columns: [[f32; 3]; 3]) -> Direction<3> {
    Direction::from_columns([
        Vector::new([
            columns[0][0] as f64,
            columns[0][1] as f64,
            columns[0][2] as f64,
        ]),
        Vector::new([
            columns[1][0] as f64,
            columns[1][1] as f64,
            columns[1][2] as f64,
        ]),
        Vector::new([
            columns[2][0] as f64,
            columns[2][1] as f64,
            columns[2][2] as f64,
        ]),
    ])
}

fn centered_half_offset(
    direction: Direction<3>,
    spacing: Spacing<3>,
    dims_xyz: [usize; 3],
) -> Vector<3> {
    let half_dim = Vector::new([
        (dims_xyz[0] as f64 - 1.0) / 2.0,
        (dims_xyz[1] as f64 - 1.0) / 2.0,
        (dims_xyz[2] as f64 - 1.0) / 2.0,
    ]);
    let scaled_half = Vector::new([
        spacing[0] * half_dim[0],
        spacing[1] * half_dim[1],
        spacing[2] * half_dim[2],
    ]);

    direction * scaled_half
}
