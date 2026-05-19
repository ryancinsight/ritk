// Maximum Intensity Projection (MIP) compute shader for GPU volume rendering.
//
// Reads a scalar volume stored as a flat f32 array in [depth, rows, cols]
// row-major order.  For each output pixel (row, col) the shader iterates along
// the depth axis and writes the maximum encountered intensity to `mip_out`.
//
// Dispatch: ceil(cols / 8) × ceil(rows / 8) × 1 workgroups.
// The fast-varying thread dimension is `id.x` = col, enabling coalesced reads
// within a warp (adjacent threads access consecutive memory locations).

struct RenderParams {
    depth: u32,
    rows:  u32,
    cols:  u32,
    _pad:  u32,
}

@group(0) @binding(0) var<storage, read>       volume:  array<f32>;
@group(0) @binding(1) var<storage, read_write> mip_out: array<f32>;
@group(0) @binding(2) var<uniform>             params:  RenderParams;

@compute @workgroup_size(8, 8, 1)
fn mip_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;

    if col >= params.cols || row >= params.rows {
        return;
    }

    // f32 minimum approximation — all realistic voxel values exceed this.
    var max_val: f32 = -3.0e38;

    let stride_d: u32  = params.rows * params.cols;
    let row_offset: u32 = row * params.cols;

    for (var d: u32 = 0u; d < params.depth; d = d + 1u) {
        let v = volume[d * stride_d + row_offset + col];
        if v > max_val {
            max_val = v;
        }
    }

    mip_out[row * params.cols + col] = max_val;
}
