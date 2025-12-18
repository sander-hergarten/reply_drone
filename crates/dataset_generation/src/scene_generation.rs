use std::path::Path;

use crate::{RngResource, SpawnRange, Textures};
use bevy::{asset::AssetPath, prelude::*};
use bevy_rapier3d::prelude::*;
use rand::Rng;

pub fn generate_scene(
    meshes: ResMut<Assets<Mesh>>,
    materials: ResMut<Assets<StandardMaterial>>,
    mut rng: ResMut<RngResource>,
    spawn_range: Res<SpawnRange>,
    textures: ResMut<Textures>,
    assets: Res<AssetServer>,
) -> Vec<(Handle<Mesh>, Handle<StandardMaterial>, Transform, Collider)> {
    let count: u32 = rng.rng.random_range(1..100);
    // let count: u32 = 2000;
    let meshes = generate_meshes_with_colliders(count, meshes, &mut rng);
    let materials = generate_materials(count, materials, textures, &mut rng, assets);
    let transforms = generate_transforms(count, &mut rng, spawn_range);

    transforms
        .into_iter()
        .zip(materials)
        .zip(meshes)
        .map(|((transform, material), (mesh, collider))| (mesh, material, transform, collider))
        .collect()
}

fn generate_meshes_with_colliders(
    count: u32,
    mut meshes: ResMut<Assets<Mesh>>,
    rng: &mut RngResource,
) -> Vec<(Handle<Mesh>, Collider)> {
    (0..count)
        .map(|_| {
            let (mesh, collider) = random_cuboid_with_collider(rng);
            (meshes.add(mesh), collider)
        })
        .collect()
}

fn generate_materials(
    count: u32,
    mut materials: ResMut<Assets<StandardMaterial>>,
    textures: ResMut<Textures>,
    rng: &mut RngResource,
    assets: Res<AssetServer>,
) -> Vec<Handle<StandardMaterial>> {
    let materials: Vec<Handle<StandardMaterial>> = (0..count)
        .map(|_| {
            let texture_handle =
                assets.load(AssetPath::from_path(&Path::new(textures.pick_random(rng))));
            StandardMaterial {
                base_color_texture: Some(texture_handle),
                ..default()
            }
        })
        .map(|material| materials.add(material))
        .collect();

    materials
}

fn generate_transforms(
    count: u32,
    rng: &mut RngResource,
    spawn_range: Res<SpawnRange>,
) -> Vec<Transform> {
    (0..count)
        .map(|_| {
            Transform::from_translation(Vec3::new(
                rng.rng.random_range(spawn_range.min.x..spawn_range.max.x),
                rng.rng.random_range(spawn_range.min.y..spawn_range.max.y),
                rng.rng.random_range(spawn_range.min.z..spawn_range.max.z),
            ))
            .with_rotation(Quat::from_euler(
                EulerRot::XYZ,
                rng.rng.random_range(0.0..2.0 * std::f32::consts::PI),
                rng.rng.random_range(0.0..2.0 * std::f32::consts::PI),
                rng.rng.random_range(0.0..2.0 * std::f32::consts::PI),
            ))
        })
        .collect()
}

pub fn random_cuboid_with_collider(rng: &mut RngResource) -> (Mesh, Collider) {
    // 1. Random Size
    // Define ranges for width, height, depth
    let width = rng.rng.random_range(0.5..3.0);
    let height = rng.rng.random_range(0.5..3.0);
    let depth = rng.rng.random_range(0.5..3.0);
    (
        Mesh::from(Cuboid::new(width, height, depth)),
        Collider::cuboid(width / 2.0, height / 2.0, depth / 2.0),
    )
}
