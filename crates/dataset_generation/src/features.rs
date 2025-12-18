use crate::{FeaturesSpawned, RngResource};
use bevy::camera::visibility::RenderLayers;
use bevy::mesh::{Indices, VertexAttributeValues};
use bevy::prelude::*;
use rand::Rng;

const FEATURES_TO_SPAWN: u16 = 1000;

#[derive(Message)]
pub struct SpawnFeatures {
    point: Vec3,
    normal: Vec3,
}

pub fn spawn_features(
    mut commands: Commands,
    mut features_spawned: ResMut<FeaturesSpawned>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    asset_server: Res<AssetServer>,
    mut features_to_spawn: MessageReader<SpawnFeatures>,
) {
    if features_spawned.0 > FEATURES_TO_SPAWN {
        // capture_image_event.send(CaptureImageEvent);
        return;
    }
    for feature_to_spawn in features_to_spawn.read() {
        let hit_point = feature_to_spawn.point;
        let normal = feature_to_spawn.normal;

        let plane_mesh = meshes.add(Mesh::from(Plane3d::new(Vec3::Y, Vec2::new(0.3, 0.3))));
        let texture_handle = asset_server.load("features/1.jpg");
        let plane_material = materials.add(StandardMaterial {
            base_color_texture: Some(texture_handle),
            ..default()
        });

        commands.spawn((
            Mesh3d(plane_mesh.clone()),
            MeshMaterial3d(plane_material),
            Transform {
                translation: hit_point + normal * 0.02,
                rotation: Quat::from_rotation_arc(Vec3::Y, normal),
                scale: Vec3::new(1.0, 1.0, 1.0),
            },
            RenderLayers::layer(0),
        ));
        commands.spawn((
            Mesh3d(plane_mesh),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                unlit: true,
                ..default()
            })),
            Transform {
                translation: hit_point + normal * 0.02,
                rotation: Quat::from_rotation_arc(Vec3::Y, normal),
                scale: Vec3::new(1.0, 1.0, 1.0),
            },
            RenderLayers::layer(1),
        ));

        features_spawned.0 += 1;
    }
}

pub fn sample_random_point_system(
    mesh_query: Query<(&Mesh3d, &GlobalTransform)>,
    meshes: ResMut<Assets<Mesh>>,
    mut request_feature_spawns: MessageWriter<SpawnFeatures>,
    mut rng: ResMut<RngResource>,
    features_spawned: ResMut<FeaturesSpawned>,
) {
    if features_spawned.0 > FEATURES_TO_SPAWN {
        return;
    }
    // 1. Calculate Total Scene Area and build a "CDF" (Cumulative Distribution Function) on the fly.
    // In a real app, you should cache these areas rather than recalculating every frame.
    let mut total_area = 0.0;

    // We need to store candidate meshes to pick one later
    struct MeshCandidate<'a> {
        mesh: &'a Mesh,
        transform: &'a GlobalTransform,
        area: f32,
        cumulative_start: f32,
    }

    let mut candidates = Vec::new();

    for (handle, transform) in mesh_query.iter() {
        if let Some(mesh) = meshes.get(handle) {
            let mesh_area = calculate_mesh_area(mesh, transform);
            if mesh_area > 0.0 {
                candidates.push(MeshCandidate {
                    mesh,
                    transform,
                    area: mesh_area,
                    cumulative_start: total_area,
                });
                total_area += mesh_area;
            }
        }
    }

    if total_area == 0.0 {
        return;
    }

    let target_area_samples = (0..FEATURES_TO_SPAWN)
        .map(|_| rng.rng.random_range(0.0..total_area))
        .collect::<Vec<_>>();

    let features_to_spawn = target_area_samples
        .into_iter()
        .filter_map(|target_area_sample: f32| {
            if let Some(chosen) = candidates.iter().find(|c| {
                target_area_sample >= c.cumulative_start
                    && target_area_sample < c.cumulative_start + c.area
            }) {
                // The specific area value relative to this mesh
                let local_sample_area = target_area_sample - chosen.cumulative_start;

                // 4. Sample the specific triangle within this mesh

                sample_point_on_mesh(chosen.mesh, chosen.transform, &mut rng, local_sample_area)
            } else {
                None
            }
        })
        .map(|(point, normal)| SpawnFeatures { point, normal });
    request_feature_spawns.write_batch(features_to_spawn);
}

fn calculate_mesh_area(mesh: &Mesh, transform: &GlobalTransform) -> f32 {
    let Some(indices) = get_indices(mesh) else {
        return 0.0;
    };
    let Some(positions) = get_positions(mesh) else {
        return 0.0;
    };

    // Extract scale from GlobalTransform to ensure area is correct in world space
    let (_scale, _, _) = transform.to_scale_rotation_translation();
    // Simplified uniform scale assumption for area calc (scale.x * scale.y approx)
    // For non-uniform scale, you must transform vertices first.
    // We will transform vertices below for accuracy.
    let matrix = transform.to_matrix();

    let mut area = 0.0;
    for i in (0..indices.len()).step_by(3) {
        let i0 = indices[i];
        let i1 = indices[i + 1];
        let i2 = indices[i + 2];

        let v0 = matrix.transform_point3(positions[i0]);
        let v1 = matrix.transform_point3(positions[i1]);
        let v2 = matrix.transform_point3(positions[i2]);

        // Triangle area = 0.5 * |(v1 - v0) x (v2 - v0)|
        area += 0.5 * (v1 - v0).cross(v2 - v0).length();
    }
    area
}

/// Picks a triangle based on weight and samples a point/normal from it
fn sample_point_on_mesh(
    mesh: &Mesh,
    transform: &GlobalTransform,
    rng: &mut RngResource,
    mut target_area: f32,
) -> Option<(Vec3, Vec3)> {
    let indices = get_indices(mesh)?;
    let positions = get_positions(mesh)?;
    let normals = get_normals(mesh)?; // Returns None if mesh has no normals

    let matrix = transform.to_matrix();
    // For normals, we need the inverse transpose matrix if there is non-uniform scaling
    let normal_matrix = matrix.inverse().transpose();

    for i in (0..indices.len()).step_by(3) {
        let i0 = indices[i] as usize;
        let i1 = indices[i + 1] as usize;
        let i2 = indices[i + 2] as usize;

        // Transform positions to world space to check area
        let v0 = matrix.transform_point3(positions[i0]);
        let v1 = matrix.transform_point3(positions[i1]);
        let v2 = matrix.transform_point3(positions[i2]);

        let area = 0.5 * (v1 - v0).cross(v2 - v0).length();

        // If our random sample falls inside this triangle's "area bucket"
        if target_area <= area {
            // Sample this triangle!

            // Random Barycentric Coordinates
            // Source: "Shape Distributions" (Osada et al.)
            let r1: f32 = rng.rng.random();
            let r2: f32 = rng.rng.random();
            let sqrt_r1 = r1.sqrt();

            let u = 1.0 - sqrt_r1;
            let v = sqrt_r1 * (1.0 - r2);
            let w = sqrt_r1 * r2; // w = 1 - u - v

            // Interpolate Position
            let random_pos = v0 * u + v1 * v + v2 * w;

            // Interpolate Normal (smooth shading)
            // Transform local normals to world space first
            let n0 = normal_matrix.transform_vector3(normals[i0]);
            let n1 = normal_matrix.transform_vector3(normals[i1]);
            let n2 = normal_matrix.transform_vector3(normals[i2]);

            let random_normal = (n0 * u + n1 * v + n2 * w).normalize();

            return Some((random_pos, random_normal));
        }

        target_area -= area;
    }

    None
}

// --- Helper accessors for Mesh data ---

fn get_positions(mesh: &Mesh) -> Option<Vec<Vec3>> {
    match mesh.attribute(Mesh::ATTRIBUTE_POSITION)? {
        VertexAttributeValues::Float32x3(ven) => {
            Some(ven.iter().map(|v| Vec3::from_array(*v)).collect::<Vec<_>>())
        }
        _ => None,
    }
}

fn get_normals(mesh: &Mesh) -> Option<Vec<Vec3>> {
    match mesh.attribute(Mesh::ATTRIBUTE_NORMAL)? {
        VertexAttributeValues::Float32x3(ven) => {
            Some(ven.iter().map(|v| Vec3::from_array(*v)).collect::<Vec<_>>())
        }

        _ => None,
    }
}

fn get_indices(mesh: &Mesh) -> Option<Vec<usize>> {
    match mesh.indices() {
        Some(Indices::U16(indices)) => Some(indices.iter().map(|&i| i as usize).collect()),
        Some(Indices::U32(indices)) => Some(indices.iter().map(|&i| i as usize).collect()),
        None => None, // This example assumes indexed meshes; non-indexed would handle 0,1,2, 3,4,5...
    }
}
