use crate::components::*;
use crate::image_capture::CaptureImageEvent;
use crate::{FeaturesSpawned, RngResource, SpawnRange};
use bevy::math::Affine3A;
use bevy::render::view::RenderLayers;
use bevy::{prelude::*, render::mesh::MeshAabb};
use bevy_rapier3d::prelude::*;
use rand::Rng;

struct DimensionVectors {
    x: Vec3,
    y: Vec3,
    z: Vec3,
}

impl DimensionVectors {
    fn apply_transform(&self, transform: &Affine3A) -> Self {
        Self {
            x: transform.transform_point3(self.x),
            y: transform.transform_point3(self.y),
            z: transform.transform_point3(self.z),
        }
    }
}

fn intersecting(
    point: Vec3,
    meshes: &ResMut<Assets<Mesh>>,
    cuboids: &Query<(&Transform, &Mesh3d), With<IsCuboid>>,
) -> bool {
    for (transform, mesh) in cuboids.iter() {
        let mesh_handle = mesh.0.clone();

        // Use 'match' to handle the Option<&Mesh> from meshes.get()
        let mesh = meshes.get(mesh_handle.id()).unwrap();
        let aabb = mesh.compute_aabb().unwrap();
        let dimensions: Vec3 = (aabb.half_extents * 2.0).into();
        let min = aabb.min().into();

        let transformation_matrix = transform.compute_affine();

        let dimension_vectors: DimensionVectors = DimensionVectors {
            x: Vec3::new(dimensions.x, 0.0, 0.0),
            y: Vec3::new(0.0, dimensions.y, 0.0),
            z: Vec3::new(0.0, 0.0, dimensions.z),
        };

        let transformed_dimension_vectors =
            dimension_vectors.apply_transform(&transformation_matrix);

        let transformed_min = transformation_matrix.transform_point3(min);

        if transformed_min.x < point.x && transformed_min.y < point.y && transformed_min.z < point.z
        {
            return ((transformed_min + transformed_dimension_vectors.x).x > point.x)
                && ((transformed_min + transformed_dimension_vectors.y).y > point.y)
                && ((transformed_min + transformed_dimension_vectors.z).z > point.z);
        }
    }
    false
}

fn find_position_and_rotation(
    rng: &mut ResMut<RngResource>,
    meshes: &ResMut<Assets<Mesh>>,
    cuboids: &Query<(&Transform, &Mesh3d), With<IsCuboid>>,
    spawn_range: &Res<SpawnRange>,
) -> (Vec3, Vec3) {
    let mut position = Vec3::new(
        rng.rng.random_range(spawn_range.min.x..spawn_range.max.x),
        rng.rng.random_range(spawn_range.min.y..spawn_range.max.y),
        rng.rng.random_range(spawn_range.min.z..spawn_range.max.z),
    );

    while intersecting(position, meshes, cuboids) {
        position = Vec3::new(
            rng.rng.random_range(spawn_range.min.x..spawn_range.max.x),
            rng.rng.random_range(spawn_range.min.y..spawn_range.max.y),
            rng.rng.random_range(spawn_range.min.z..spawn_range.max.z),
        );
    }

    let random_rotation = Quat::from_euler(
        EulerRot::XYZ,
        rng.rng.random_range(0.0..360.0),
        rng.rng.random_range(0.0..360.0),
        rng.rng.random_range(0.0..360.0),
    );
    (
        position,
        Vec3::from(random_rotation.to_euler(EulerRot::XYZ)),
    )
}

pub fn spawn_features(
    mut commands: Commands,
    mut rng: ResMut<RngResource>,
    mut features_spawned: ResMut<FeaturesSpawned>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    asset_server: Res<AssetServer>,
    cuboids: Query<(&Transform, &Mesh3d), With<IsCuboid>>,
    spawn_range: Res<SpawnRange>,
    rapier_context: ReadRapierContext,
    mut capture_image_event: EventWriter<CaptureImageEvent>,
) {
    if features_spawned.0 > 10 {
        capture_image_event.send(CaptureImageEvent);
        return;
    }

    let (ray_origin, ray_direction) =
        find_position_and_rotation(&mut rng, &meshes, &cuboids, &spawn_range);
    let ctx = rapier_context.single();

    let max_distance = 100.0;
    let solid = true;
    let filter = QueryFilter::default();
    if let Some((_entity, intersection)) =
        ctx.cast_ray_and_get_normal(ray_origin, ray_direction, max_distance, solid, filter)
    {
        println!("Intersection");
        let hit_point = intersection.point;
        let normal = intersection.normal;

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
                ..default()
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
                ..default()
            },
            RenderLayers::layer(1),
        ));

        features_spawned.0 += 1;
    }
}
