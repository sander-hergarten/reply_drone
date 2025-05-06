use bevy::{ecs::system::Resource, image::Image, render::view::screenshot::Screenshot};

use std::collections::HashMap;

pub type Position = [i32; 3];
pub type Rotation = [i32; 3];
pub type NodeId = u32;
pub type Seed = u64;

#[derive(Resource)]
pub struct Engine {
    pub node_positions: Vec<Position>,
    pub node_rotations: Vec<Rotation>,
}

#[derive(Debug)]
pub struct SingleNodeState {
    pub id: NodeId,
    pub position: Position,
    pub rotation: Rotation,
    pub depth_map: Image,
    pub postion_of_other_nodes: Vec<Position>,
    pub rotation_of_other_nodes: Vec<Rotation>,
}

pub type FullGraphState = HashMap<NodeId, SingleNodeState>;
