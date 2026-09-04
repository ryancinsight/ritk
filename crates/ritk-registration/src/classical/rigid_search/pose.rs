use crate::types::AffineTransform;

pub(super) fn euler_zyx(alpha: f64, beta: f64, gamma: f64) -> [[f64; 3]; 3] {
    let (sin_alpha, cos_alpha) = alpha.sin_cos();
    let (sin_beta, cos_beta) = beta.sin_cos();
    let (sin_gamma, cos_gamma) = gamma.sin_cos();
    let z = [
        [1.0, 0.0, 0.0],
        [0.0, cos_alpha, -sin_alpha],
        [0.0, sin_alpha, cos_alpha],
    ];
    let y = [
        [cos_beta, 0.0, sin_beta],
        [0.0, 1.0, 0.0],
        [-sin_beta, 0.0, cos_beta],
    ];
    let x = [
        [cos_gamma, -sin_gamma, 0.0],
        [sin_gamma, cos_gamma, 0.0],
        [0.0, 0.0, 1.0],
    ];
    multiply_3x3(multiply_3x3(z, y), x)
}

pub(super) fn multiply_3x3(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut product = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            product[row][column] = (0..3)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum();
        }
    }
    product
}

pub(super) fn rigid_about_centroid(
    rotation: [[f64; 3]; 3],
    fixed_centroid: [f64; 3],
    moving_centroid: [f64; 3],
) -> AffineTransform {
    let rotated_centroid = [
        rotation[0][0] * fixed_centroid[0]
            + rotation[0][1] * fixed_centroid[1]
            + rotation[0][2] * fixed_centroid[2],
        rotation[1][0] * fixed_centroid[0]
            + rotation[1][1] * fixed_centroid[1]
            + rotation[1][2] * fixed_centroid[2],
        rotation[2][0] * fixed_centroid[0]
            + rotation[2][1] * fixed_centroid[1]
            + rotation[2][2] * fixed_centroid[2],
    ];
    let translation = [
        moving_centroid[0] - rotated_centroid[0],
        moving_centroid[1] - rotated_centroid[1],
        moving_centroid[2] - rotated_centroid[2],
    ];
    AffineTransform::new([
        rotation[0][0],
        rotation[0][1],
        rotation[0][2],
        translation[0],
        rotation[1][0],
        rotation[1][1],
        rotation[1][2],
        translation[1],
        rotation[2][0],
        rotation[2][1],
        rotation[2][2],
        translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ])
}
