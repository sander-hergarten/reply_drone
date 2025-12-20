@group(0) @binding(0) var input_texture: texture_2d<f32>;

// CHANGED: r8unorm -> rgba8unorm
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

struct Params { alpha: f32 }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let dims = textureDimensions(input_texture);
    if (x >= dims.x || y >= dims.y) { return; }

    let val = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0).r;
    let result = min(1.0, params.alpha + val);

    // Store as RGBA. We only care about the Red channel (result).
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(result, 0.0, 0.0, 1.0));
}