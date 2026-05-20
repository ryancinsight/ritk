// Volume Rendering (VR) compute shader — front-to-back alpha compositing.
//
// For each output pixel (row, col) iterates the depth axis accumulating colour
// and opacity with the front-to-back compositing equation:
//
//   norm    = clamp((volume[d,r,c] - wl_lo) / wl_range, 0, 1)
//   a       = alpha_scale * norm
//   rgb     = lut[floor(norm * 255)]           (256-entry f32 RGBA colormap LUT)
//   contrib = (1 - acc_alpha) * a
//   acc_rgb += contrib * rgb
//   acc_alpha += contrib
//   early-exit when acc_alpha >= 0.99  (fully opaque; further layers contribute < 1%)
//
// # Output format
//   `vr_out`: 1 packed u32 per pixel (row-major), written via pack4x8unorm.
//   Byte layout in memory (little-endian): R G B A, each 0–255.
//   This is 4× smaller than the previous 4×f32 layout (4 bytes vs 16 bytes per
//   pixel), reducing staging buffer allocation and map/unmap cost by 4×.
//   The host reads the staging buffer as `&[u8]` and forwards directly to
//   egui::ColorImage::from_rgba_unmultiplied — zero CPU post-processing.
//
// # Dispatch
//   ceil(cols / 8) × ceil(rows / 8) × 1 workgroups.
//   Fast-varying dimension id.x = col → coalesced volume reads within each warp.

struct VrParams {
    depth       : u32,
    rows        : u32,
    cols        : u32,
    _pad0       : u32,
    wl_lo       : f32,
    wl_range    : f32,
    alpha_scale : f32,
    _pad1       : f32,
}

@group(0) @binding(0) var<storage, read>       volume : array<f32>;
@group(0) @binding(1) var<storage, read_write> vr_out : array<u32>;  // 1 packed u32 per pixel
@group(0) @binding(2) var<uniform>             params : VrParams;
@group(0) @binding(3) var<storage, read>       lut    : array<f32>;  // 256 * 4 f32 RGBA

@compute @workgroup_size(8, 8, 1)
fn vr_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;

    if col >= params.cols || row >= params.rows {
        return;
    }

    var acc_r     : f32 = 0.0;
    var acc_g     : f32 = 0.0;
    var acc_b     : f32 = 0.0;
    var acc_alpha : f32 = 0.0;

    let stride_d   : u32 = params.rows * params.cols;
    let row_offset : u32 = row * params.cols;

    for (var d: u32 = 0u; d < params.depth; d = d + 1u) {
        let v    = volume[d * stride_d + row_offset + col];
        let norm = clamp((v - params.wl_lo) / params.wl_range, 0.0, 1.0);
        let a    = params.alpha_scale * norm;

        // 256-entry RGBA LUT: 4 f32 per entry (R, G, B, A_unused).
        // lut_base ∈ [0, 255]*4 = [0, 1020]; lut[1023] is within bounds.
        let lut_base : u32 = u32(norm * 255.0) * 4u;
        let lr = lut[lut_base + 0u];
        let lg = lut[lut_base + 1u];
        let lb = lut[lut_base + 2u];

        let contrib = (1.0 - acc_alpha) * a;
        acc_r     += contrib * lr;
        acc_g     += contrib * lg;
        acc_b     += contrib * lb;
        acc_alpha += contrib;

        // Early exit: pixel is fully opaque; remaining layers contribute < 1%.
        if acc_alpha >= 0.99 {
            break;
        }
    }

    // Pack composited RGBA into one u32 (little-endian: R G B A in memory).
    // pack4x8unorm clamps each component to [0,1] and rounds to nearest u8.
    vr_out[row * params.cols + col] = pack4x8unorm(
        vec4<f32>(acc_r, acc_g, acc_b, acc_alpha)
    );
}
