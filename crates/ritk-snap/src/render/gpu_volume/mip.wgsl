// Maximum Intensity Projection (MIP) compute shader for GPU volume rendering.
//
// Reads a scalar volume stored as a flat f32 array in [depth, rows, cols]
// row-major order.  For each output pixel (row, col) the shader iterates along
// the depth axis, finds the maximum intensity, applies WL normalisation and
// the 256-entry colormap LUT in-shader, and writes a packed u32 RGBA pixel to
// `mip_out` — eliminating all post-readback CPU processing.
//
// # Bindings
//   0: volume   — flat f32 array [depth * rows * cols]
//   1: mip_out  — packed u32 RGBA, one element per output pixel (row-major)
//   2: params   — RenderParams uniform (shape + DICOM presentation)
//   3: lut      — 256-entry f32 RGBA colormap LUT (1 024 f32 values)
//
// # Output packing
//   pack4x8unorm(vec4<f32>(r, g, b, 1.0)) stores the pixel as four u8 values
//   in little-endian order: byte 0 = R, byte 1 = G, byte 2 = B, byte 3 = A.
//   On the host side the staging buffer can be reinterpreted directly as
//   `&[u8]` and passed to egui::ColorImage::from_rgba_unmultiplied without
//   any CPU post-processing.
//
// # DICOM presentation (in-shader)
//   presentation bits 0–1 select LINEAR, LINEAR_EXACT, or SIGMOID; bit 2
//   applies the MONOCHROME1 inversion after the VOI function.
//   lut_base = floor(norm * 255.0) * 4
//
// # Dispatch
//   ceil(cols / 8) × ceil(rows / 8) × 1 workgroups.
//   Fast-varying dimension id.x = col — coalesced volume reads within a warp.

struct RenderParams {
    depth    : u32,
    rows     : u32,
    cols     : u32,
    _pad0    : u32,
    center       : f32,
    width        : f32,
    presentation : u32,
    _pad2        : u32,
}

@group(0) @binding(0) var<storage, read>       volume  : array<f32>;
@group(0) @binding(1) var<storage, read_write> mip_out : array<u32>;
@group(0) @binding(2) var<uniform>             params  : RenderParams;
@group(0) @binding(3) var<storage, read>       lut     : array<f32>;

fn window_norm(value: f32) -> f32 {
    // NaN has no ordered DICOM display value; match the CPU presentation
    // path's lower endpoint while preserving +/-infinity endpoint behavior.
    if value != value {
        return 0.0;
    }
    let function = params.presentation & 3u;
    var norm: f32;
    if function == 0u {
        let lower = params.center - 0.5 - (params.width - 1.0) * 0.5;
        let upper = params.center - 0.5 + (params.width - 1.0) * 0.5;
        if value <= lower {
            norm = 0.0;
        } else if value > upper {
            norm = 1.0;
        } else {
            norm = (value - (params.center - 0.5)) / (params.width - 1.0) + 0.5;
        }
    } else if function == 1u {
        let lower = params.center - 0.5 * params.width;
        let upper = params.center + 0.5 * params.width;
        if value <= lower {
            norm = 0.0;
        } else if value > upper {
            norm = 1.0;
        } else {
            norm = (value - params.center) / params.width + 0.5;
        }
    } else {
        norm = 1.0 / (1.0 + exp(4.0 * (params.center - value) / params.width));
    }
    norm = clamp(norm, 0.0, 1.0);
    if (params.presentation & 4u) != 0u {
        norm = 1.0 - norm;
    }
    return norm;
}

@compute @workgroup_size(8, 8, 1)
fn mip_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;

    if col >= params.cols || row >= params.rows {
        return;
    }

    // f32 minimum approximation — all realistic voxel values exceed this.
    var max_val : f32 = -3.0e38;

    let stride_d   : u32 = params.rows * params.cols;
    let row_offset : u32 = row * params.cols;

    for (var d: u32 = 0u; d < params.depth; d = d + 1u) {
        let v = volume[d * stride_d + row_offset + col];
        if v > max_val {
            max_val = v;
        }
    }

    // WL normalisation + colormap LUT lookup, all in shader.
    // lut_base ∈ [0, 255] * 4 = [0, 1020]; lut[1023] is within the 1024-entry array.
    let norm     : f32 = window_norm(max_val);
    let lut_base : u32 = u32(norm * 255.0) * 4u;
    let r : f32 = lut[lut_base + 0u];
    let g : f32 = lut[lut_base + 1u];
    let b : f32 = lut[lut_base + 2u];

    // Pack RGB + full alpha into one u32 (little-endian: R G B A in memory).
    mip_out[row * params.cols + col] = pack4x8unorm(vec4<f32>(r, g, b, 1.0));
}
