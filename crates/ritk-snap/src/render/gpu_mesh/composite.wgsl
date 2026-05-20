// OIT composite + SSAO application compute shader.
//
// Reads the N_LAYERS Rgba8Unorm color layers produced by depth peeling and
// the AO factor buffer produced by the SSAO pass, then composites them
// back-to-front (Porter-Duff "over" operator) and writes packed u32 RGBA to
// the readback output buffer.
//
// # OIT compositing (back-to-front Porter-Duff "over")
//
//   acc = (0,0,0,0)   ← transparent background
//   for layer = N_LAYERS-1 downto 0:
//     let c = color_layers[coord, layer]
//     acc.rgb = c.rgb * c.a + acc.rgb * (1 - c.a)
//     acc.a   = c.a + acc.a * (1 - c.a)
//
// # SSAO application
//
//   SSAO is applied only to layer 0 (the nearest visible surface), which is
//   where per-pixel ambient occlusion has the highest perceptual impact.
//   AO factor multiplies the layer-0 RGB to attenuate ambient contribution.
//   Layers 1..N−1 carry their pre-baked Phong color without modification.
//
// # Output packing
//
//   pack4x8unorm(final_color) → u32 (R=byte0, G=byte1, B=byte2, A=byte3)
//   Host reads staging buffer directly as &[u8] → egui::ColorImage.
//
// Bindings (group 0):
//   0: color_layers — texture_2d_array<f32>, Rgba8Unorm, N_LAYERS array layers
//   1: ao_buf       — storage<read> array<f32>, one f32 per pixel
//   2: output       — storage<read_write> array<u32>, packed RGBA
//   3: comp_params  — uniform, CompUniforms (rows, cols)

const N_LAYERS: u32 = 4u;

struct CompUniforms {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var                      color_layers: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read>        ao_buf:       array<f32>;
@group(0) @binding(2) var<storage, read_write>  output:       array<u32>;
@group(0) @binding(3) var<uniform>              comp_params:  CompUniforms;

@compute @workgroup_size(8, 8, 1)
fn composite_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;
    if col >= comp_params.cols || row >= comp_params.rows { return; }

    let coord   = vec2<i32>(i32(col), i32(row));
    let ao      = ao_buf[row * comp_params.cols + col];

    // Back-to-front compositing: start from last layer (furthest) down to layer 0.
    var acc = vec4<f32>(0.0);

    var layer = N_LAYERS;
    loop {
        if layer == 0u { break; }
        layer -= 1u;

        let c = textureLoad(color_layers, coord, layer, 0);
        // Transparent pixels (alpha ≈ 0) contribute nothing.
        if c.a < 0.001 { continue; }

        // Apply SSAO only to the nearest surface (layer 0).
        var rgb = c.rgb;
        if layer == 0u {
            rgb = c.rgb * ao;
        }

        // Porter-Duff "over": dst = src + dst * (1 - src.alpha)
        acc = vec4<f32>(
            rgb * c.a + acc.rgb * (1.0 - c.a),
            c.a + acc.a * (1.0 - c.a),
        );
    }

    let final_color = clamp(acc, vec4<f32>(0.0), vec4<f32>(1.0));
    output[row * comp_params.cols + col] = pack4x8unorm(final_color);
}
