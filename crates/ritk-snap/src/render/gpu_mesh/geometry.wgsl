// Base geometry pass (peel layer 0): vertex transform + Phong shading.
//
// Outputs TWO color attachments:
//   @location(0) color       → Rgba8Unorm  (Phong-shaded RGBA)
//   @location(1) normal_depth → Rgba16Float (view-space normal encoded to [0,1] in xyz,
//                                            linear depth = -view_pos.z in w)
//
// Bindings (group 0):
//   0: SceneUniforms   — mvp (64), mv (64), peel_pass (4), pad (12) = 144 bytes
//   1: LightBlock      — 2 × LightUniform = 96 bytes
//   2: MaterialUniforms — 3 × vec4 = 48 bytes
//
// NOTE: peel_pass is always 0 for this shader; the field is read but does not
//       alter behavior here (included for uniform-buffer layout compatibility).

// ── Struct declarations ───────────────────────────────────────────────────────

struct SceneUniforms {
    mvp:       mat4x4<f32>,   // offset   0, 64 bytes
    mv:        mat4x4<f32>,   // offset  64, 64 bytes
    peel_pass: u32,            // offset 128,  4 bytes
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
}

struct LightUniform {
    direction_view: vec3<f32>,   // align 16, offset  0
    _pad0:          f32,
    color:          vec3<f32>,   // align 16, offset 16
    _pad1:          f32,
    ambient:        vec3<f32>,   // align 16, offset 32
    _pad2:          f32,
}

struct LightBlock {
    lights: array<LightUniform, 2>,
}

struct MaterialUniforms {
    diffuse:        vec4<f32>,   // RGBA diffuse
    specular_shine: vec4<f32>,   // xyz = specular, w = shininess
    opacity_pad:    vec4<f32>,   // x = opacity
}

// ── Bindings ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> scene:    SceneUniforms;
@group(0) @binding(1) var<uniform> lights:   LightBlock;
@group(0) @binding(2) var<uniform> material: MaterialUniforms;

// ── Vertex stage ──────────────────────────────────────────────────────────────

struct VertexIn {
    @location(0) position: vec4<f32>,   // xyz = world pos, w = 1.0
    @location(1) normal:   vec4<f32>,   // xyz = world normal, w = 0.0
}

struct VertexOut {
    @builtin(position) clip_pos:    vec4<f32>,
    @location(0)       view_pos:    vec3<f32>,
    @location(1)       view_normal: vec3<f32>,
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = scene.mvp * in.position;

    let view_pos4  = scene.mv * in.position;
    out.view_pos   = view_pos4.xyz / view_pos4.w;

    // Upper-left 3×3 of MV for normal transform (assumes uniform scaling).
    let mv3 = mat3x3<f32>(
        scene.mv[0].xyz,
        scene.mv[1].xyz,
        scene.mv[2].xyz,
    );
    out.view_normal = normalize(mv3 * in.normal.xyz);
    return out;
}

// ── Fragment stage ────────────────────────────────────────────────────────────

struct FragOut {
    @location(0) color:        vec4<f32>,   // → Rgba8Unorm
    @location(1) normal_depth: vec4<f32>,   // → Rgba16Float
}

/// Phong illumination model:
/// I = k_a·I_a + Σ_i [ k_d·max(n·l_i,0)·I_i + k_s·max(r_i·v,0)^s·I_i ]
@fragment
fn fs_base_main(in: VertexOut) -> FragOut {
    let n = normalize(in.view_normal);
    let v = normalize(-in.view_pos);   // view direction (toward camera)

    var color = vec3<f32>(0.0);

    for (var i: i32 = 0; i < 2; i++) {
        let l = normalize(lights.lights[i].direction_view);

        // Ambient
        color += lights.lights[i].ambient * material.diffuse.rgb;

        // Diffuse: k_d · max(n·l, 0) · I_light
        let n_dot_l = max(dot(n, l), 0.0);
        color += material.diffuse.rgb * n_dot_l * lights.lights[i].color;

        // Specular: k_s · max(r·v, 0)^s · I_light  (Phong reflection)
        if n_dot_l > 0.0 {
            let r     = reflect(-l, n);
            let r_dot_v = max(dot(r, v), 0.0);
            let spec  = pow(r_dot_v, material.specular_shine.w);
            color += material.specular_shine.xyz * spec * lights.lights[i].color;
        }
    }

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

    var out: FragOut;
    out.color = vec4<f32>(color, material.opacity_pad.x);

    // Encode view-space normal to [0,1]; store positive linear depth in w.
    // linear_depth = -view_pos.z (positive distance from camera).
    out.normal_depth = vec4<f32>(n * 0.5 + 0.5, -in.view_pos.z);
    return out;
}
