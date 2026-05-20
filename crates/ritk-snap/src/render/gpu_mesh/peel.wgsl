// Depth-peeling geometry pass (layers 1 .. N-1): Phong shading + depth discard.
//
// Discards any fragment whose NDC depth is ≤ the corresponding depth stored in
// the previous layer's depth texture.  This extracts the next-closest surface
// layer behind the already-rendered layers.
//
// Output: ONE color attachment:
//   @location(0) color → Rgba8Unorm (Phong-shaded RGBA)
//
// Bindings (group 0):
//   0: SceneUniforms    — 144 bytes (same layout as geometry.wgsl)
//   1: LightBlock       —  96 bytes
//   2: MaterialUniforms —  48 bytes
//   3: prev_depth_tex   — texture_depth_2d (previous peel layer depth buffer)
//
// Peel discard condition:
//   frag_depth = in.clip_pos.z  ∈ [0, 1]  (NDC depth after viewport transform)
//   prev_d     = textureLoad(prev_depth_tex, pixel, 0)  ∈ [0, 1]
//   DISCARD if frag_depth <= prev_d + PEEL_EPS
//
// PEEL_EPS = 1e-4: prevents z-fighting with the immediately preceding layer.
// Analytically justified: depth32f precision at depth 0.5 ≈ 2.3e-10; 1e-4 safely
// separates layers ≥ 0.1 mm apart for standard medical near/far settings.

// ── Struct declarations (mirror geometry.wgsl layout exactly) ─────────────────

struct SceneUniforms {
    mvp:       mat4x4<f32>,
    mv:        mat4x4<f32>,
    peel_pass: u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
}

struct LightUniform {
    direction_view: vec3<f32>,
    _pad0:          f32,
    color:          vec3<f32>,
    _pad1:          f32,
    ambient:        vec3<f32>,
    _pad2:          f32,
}

struct LightBlock {
    lights: array<LightUniform, 2>,
}

struct MaterialUniforms {
    diffuse:        vec4<f32>,
    specular_shine: vec4<f32>,
    opacity_pad:    vec4<f32>,
}

// ── Bindings ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> scene:          SceneUniforms;
@group(0) @binding(1) var<uniform> lights:         LightBlock;
@group(0) @binding(2) var<uniform> material:       MaterialUniforms;
@group(0) @binding(3) var          prev_depth_tex: texture_depth_2d;

const PEEL_EPS: f32 = 1e-4;

// ── Vertex stage ──────────────────────────────────────────────────────────────

struct VertexIn {
    @location(0) position: vec4<f32>,
    @location(1) normal:   vec4<f32>,
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

    let mv3 = mat3x3<f32>(
        scene.mv[0].xyz,
        scene.mv[1].xyz,
        scene.mv[2].xyz,
    );
    out.view_normal = normalize(mv3 * in.normal.xyz);
    return out;
}

// ── Fragment stage ────────────────────────────────────────────────────────────

@fragment
fn fs_peel_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Depth peel discard: reject fragment if it is at/in-front-of the previous layer.
    let px       = vec2<i32>(i32(in.clip_pos.x), i32(in.clip_pos.y));
    let prev_d   = textureLoad(prev_depth_tex, px, 0);
    if in.clip_pos.z <= prev_d + PEEL_EPS {
        discard;
    }

    // Phong shading (identical to geometry.wgsl fs_base_main)
    let n = normalize(in.view_normal);
    let v = normalize(-in.view_pos);

    var color = vec3<f32>(0.0);

    for (var i: i32 = 0; i < 2; i++) {
        let l = normalize(lights.lights[i].direction_view);
        color += lights.lights[i].ambient * material.diffuse.rgb;
        let n_dot_l = max(dot(n, l), 0.0);
        color += material.diffuse.rgb * n_dot_l * lights.lights[i].color;
        if n_dot_l > 0.0 {
            let r       = reflect(-l, n);
            let r_dot_v = max(dot(r, v), 0.0);
            let spec    = pow(r_dot_v, material.specular_shine.w);
            color += material.specular_shine.xyz * spec * lights.lights[i].color;
        }
    }

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(color, material.opacity_pad.x);
}
