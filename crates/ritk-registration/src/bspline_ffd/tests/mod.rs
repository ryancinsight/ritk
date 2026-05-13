pub(super) mod basis;
pub(super) mod integration;
pub(super) mod metric;
pub(super) mod pyramid;
pub(super) mod regularization;
pub(super) mod warp;

/// Smooth 3D test image: `I[z,y,x] = sin(π z/nz) · cos(π y/ny) · (x + 1)`.
pub(super) fn make_test_image(dims: [usize; 3]) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    (0..nz * ny * nx)
        .map(|fi| {
            let ix = fi % nx;
            let iy = (fi / nx) % ny;
            let iz = fi / (ny * nx);
            let sz = std::f32::consts::PI * iz as f32 / nz as f32;
            let sy = std::f32::consts::PI * iy as f32 / ny as f32;
            sz.sin() * sy.cos() * (ix as f32 + 1.0)
        })
        .collect()
}
