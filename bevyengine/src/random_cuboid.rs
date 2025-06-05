use crate::RngResource;
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use rand::prelude::*;

// New function to spawn a random cuboid

fn apply_random_texture(
    rng: &mut RngResource,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    asset_server: &Res<AssetServer>,
) -> Handle<StandardMaterial> {
    // List of available textures in the assets/textures folder
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

    // Select a random texture
    let texture_path = &textures[rng.rng.random_range(0..textures.len())];

    // Load the texture as an asset
    let texture_handle = asset_server.load(texture_path);

    // Create a material with the texture
    materials.add(StandardMaterial {
        base_color_texture: Some(texture_handle),
        ..default()
    })
}
