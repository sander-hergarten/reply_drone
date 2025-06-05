use crate::RngResource;
use bevy::prelude::*;
use rand::Rng;

pub fn generate_scene(
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut rng: ResMut<RngResource>,
    assets: Res<AssetServer>,
) -> Vec<(Handle<Mesh>, Handle<StandardMaterial>, Transform)> {
    let count: u32 = rng.rng.random_range(1..100);
    // let count: u32 = 2000;
    let meshes = generate_meshes(count, meshes, &mut rng);
    let materials = generate_materials(count, assets, materials, &mut rng);
    let transforms = generate_transforms(count, &mut rng);

    let scene = transforms
        .into_iter()
        .zip(materials)
        .zip(meshes)
        .map(|((transform, material), mesh)| (mesh, material, transform))
        .collect();

    scene
}

fn generate_meshes(
    count: u32,
    mut meshes: ResMut<Assets<Mesh>>,
    rng: &mut RngResource,
) -> Vec<Handle<Mesh>> {
    (0..count).map(|_| meshes.add(random_cuboid(rng))).collect()
}

fn generate_materials(
    count: u32,
    assets: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    rng: &mut RngResource,
) -> Vec<Handle<StandardMaterial>> {
    let textures: Vec<String> = std::fs::read_dir("assets/textures")
        .unwrap()
        .filter_map(|entry| {
            let path = entry.unwrap().path();
            if path.is_file() {
                path.strip_prefix("assets")
                    .ok()
                    .and_then(|p| p.to_str())
                    .map(|s| s.to_string())
            } else {
                None
            }
        })
        .collect();

    let materials: Vec<Handle<StandardMaterial>> = (0..count)
        .map(|_| {
            let texture_path = &textures[rng.rng.random_range(0..textures.len())];
            let texture_handle = assets.load(texture_path);
            StandardMaterial {
                base_color_texture: Some(texture_handle),
                ..default()
            }
        })
        .map(|material| materials.add(material))
        .collect();

    materials
}

fn generate_transforms(count: u32, rng: &mut RngResource) -> Vec<Transform> {
    (0..count)
        .map(|_| {
            Transform::from_xyz(
                rng.rng.random_range(-10.0..10.0),
                rng.rng.random_range(-10.0..10.0),
                rng.rng.random_range(-10.0..10.0),
            )
        })
        .collect()
}

pub fn random_cuboid(rng: &mut RngResource) -> Mesh {
    // 1. Random Size
    // Define ranges for width, height, depth
    let width = rng.rng.random_range(0.5..3.0);
    let height = rng.rng.random_range(0.5..3.0);
    let depth = rng.rng.random_range(0.5..3.0);
    Mesh::from(Cuboid::new(width, height, depth))
}
