// Screen-Space Ambient Occlusion (SSAO) compute shader.
//
// Reads the G-buffer produced by the base geometry pass (normal_depth_tex,
// Rgba16Float: xyz = view-space normal encoded to [0,1], w = linear depth)
// and writes a per-pixel AO factor to ao_buf (f32 per pixel, value in [0,1]).
//
// # Algorithm
//
// For each pixel with geometry (linear_depth > 0):
//  1. Reconstruct view-space position from NDC + linear depth.
//  2. Decode view-space surface normal.
//  3. Build TBN matrix (tangent, bitangent, normal) via Gram-Schmidt.
//  4. For each of the N_SAMPLES hemisphere kernel samples:
//     a. Rotate sample from tangent space to view space via TBN.
//     b. Project rotated sample to screen space.
//     c. Sample the geometry depth at that screen position.
//     d. If geometry is closer than the sample point → occluded.
//     e. Weight occlusion by range factor (suppress far-surface artefacts).
//  5. AO = 1 - (weighted_occlusion / N_SAMPLES) * strength
//
// # View-space reconstruction (right-handed, camera looks along -Z)
//
//   focal_x = proj[0] = f / aspect,  focal_y = proj[5] = f
//   ndc_x = 2*(col+0.5)/cols - 1
//   ndc_y = 1 - 2*(row+0.5)/rows    (screen Y down, NDC Y up → flip)
//   view_pos = (ndc_x * ld / focal_x,  ndc_y * ld / focal_y,  -ld)
//   where ld = linear_depth = -view_pos.z (positive)
//
// Bindings (group 0):
//   0: SsaoUniforms         — 48 bytes
//   1: normal_depth_tex     — texture_2d<f32>, Rgba16Float G-buffer
//   2: ao_buf               — storage<read_write> array<f32>
//   3: kernel               — storage<read> array<vec4<f32>>, 16 entries

struct SsaoUniforms {
    near:       f32,
    far:        f32,
    focal_x:    f32,
    focal_y:    f32,
    radius:     f32,
    bias:       f32,
    n_samples:  u32,
    strength:   f32,
    viewport_w: u32,
    viewport_h: u32,
    _pad0:      u32,
    _pad1:      u32,
}

@group(0) @binding(0) var<uniform>             ssao:         SsaoUniforms;
@group(0) @binding(1) var                       normal_depth: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write>  ao_buf:       array<f32>;
@group(0) @binding(3) var<storage, read>        kernel:       array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn ssao_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;
    if col >= ssao.viewport_w || row >= ssao.viewport_h { return; }

    let buf_idx = row * ssao.viewport_w + col;
    let coord   = vec2<i32>(i32(col), i32(row));
    let nd      = textureLoad(normal_depth, coord, 0);

    let linear_depth = nd.w;
    // Background pixel: no geometry → no occlusion.
    if linear_depth <= 0.0 {
        ao_buf[buf_idx] = 1.0;
        return;
    }

    // Decode view-space normal from [0,1] encoding.
    let view_normal = normalize(nd.xyz * 2.0 - 1.0);

    // Reconstruct view-space position.
    let ndc_x   = (f32(col) + 0.5) / f32(ssao.viewport_w) * 2.0 - 1.0;
    let ndc_y   = 1.0 - (f32(row) + 0.5) / f32(ssao.viewport_h) * 2.0;
    let view_pos = vec3<f32>(
        ndc_x * linear_depth / ssao.focal_x,
        ndc_y * linear_depth / ssao.focal_y,
        -linear_depth,
    );

    // Build TBN: tangent frame oriented along surface normal.
    // Fallback for normals nearly parallel to world-up: use world-X as reference.
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if abs(view_normal.y) > 0.99 {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent   = normalize(cross(up, view_normal));
    let bitangent = cross(view_normal, tangent);

    var occlusion: f32 = 0.0;

    for (var i: u32 = 0u; i < ssao.n_samples; i++) {
        let k = kernel[i].xyz;

        // Rotate kernel sample from tangent space to view space via TBN.
        let sample_vs  = tangent * k.x + bitangent * k.y + view_normal * k.z;
        let sample_pos = view_pos + sample_vs * ssao.radius;

        // Project sample to screen space.
        // sample_pos.z < 0 (view space, in front of camera).
        let sample_ld = -sample_pos.z; // positive linear depth of sample
        if sample_ld <= 0.0 { continue; }

        let s_ndc_x    = ssao.focal_x * sample_pos.x / sample_ld;
        let s_ndc_y    = ssao.focal_y * sample_pos.y / sample_ld;
        let s_screen_x = (s_ndc_x + 1.0) * 0.5;
        let s_screen_y = (1.0 - s_ndc_y) * 0.5;    // flip Y

        let sx = clamp(
            i32(s_screen_x * f32(ssao.viewport_w)),
            0, i32(ssao.viewport_w) - 1
        );
        let sy = clamp(
            i32(s_screen_y * f32(ssao.viewport_h)),
            0, i32(ssao.viewport_h) - 1
        );

        let depth_at_sample = textureLoad(normal_depth, vec2<i32>(sx, sy), 0).w;

        // Range attenuation: suppress occlusion from surfaces far from current fragment.
        // Prevents distant geometry from contributing to near-surface SSAO.
        let delta        = abs(linear_depth - depth_at_sample) + 0.001;
        let range_factor = smoothstep(0.0, 1.0, ssao.radius / delta);

        // Occluded if geometry at sample screen position is closer to camera than sample.
        if depth_at_sample > 0.0 && depth_at_sample < sample_ld - ssao.bias {
            occlusion += range_factor;
        }
    }

    let ao = 1.0 - (occlusion / f32(ssao.n_samples)) * ssao.strength;
    ao_buf[buf_idx] = clamp(ao, 0.0, 1.0);
}
