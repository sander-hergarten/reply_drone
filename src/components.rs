use bevy::ecs::component::Component;

// marker for depthview quads
#[derive(Component)]
pub struct DepthQuad;

// marker for camera and depthquad index
#[derive(Component)]
pub struct ComponentIndex(pub u32);
