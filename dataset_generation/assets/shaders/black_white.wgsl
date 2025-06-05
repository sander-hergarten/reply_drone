struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vertex(
    @location(0) vertex_position: vec3<f32>,
    @location(1) vertex_normal: vec3<f32>,
    @location(2) vertex_uv: vec2<f32>,
    @builtin(instance_index) instance_index: u32,
    @location(3) instance_position: vec3<f32>,
    @location(4) instance_rotation: vec4<f32>,
    @location(5) instance_scale: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Apply instance transform
    let rotated_position = quaternion_rotate(instance_rotation, vertex_position);
    let scaled_position = rotated_position * instance_scale;
    let world_position = scaled_position + instance_position;
    
    out.position = vec4<f32>(world_position, 1.0);
    out.world_position = world_position;
    out.normal = quaternion_rotate(instance_rotation, vertex_normal);
    
    return out;
}

@fragment
fn fragment(
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> @location(0) vec4<f32> {
    // Return pure white for features, pure black for cuboids
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}

fn quaternion_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    return v + 2.0 * (q.w * uv + uuv);
} 