@group(0) @binding(0) var input_texture: texture_2d<f32>;

// CHANGED: r8unorm -> rgba8unorm
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

struct Params { alpha: f32 }
@group(0) @binding(2) var<uniform> params: Params;

@group(0) @binding(3) var image_texture: texture_2d<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let dims = textureDimensions(input_texture);
    if (x >= dims.x || y >= dims.y) { return; }

    let val = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0).r;
    let mask_result = min(1.0, params.alpha + val);
    
    let rgb = textureLoad(image_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
    let result = rgb * mask_result;

    // Store as RGBA with RGB multiplied by mask result and alpha = 1.0
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(result, 1.0));
}