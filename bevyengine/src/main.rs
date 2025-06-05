use bevy::core_pipeline::Skybox;
use bevy::prelude::*;
use bevy_rapier3d::prelude::*; // pulls in ReadRapierContext, QueryFilter, …
use rand::prelude::*;

mod camera_controller;
mod random_cuboid;
mod scene_generation;

#[derive(Resource)]
struct RngResource {
    rng: SmallRng,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(camera_controller::CameraControllerPlugin)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        // .add_plugins(RapierDebugRenderPlugin::default())
        .insert_resource(RngResource {
            rng: SmallRng::from_rng(&mut rand::rng()), // <- simpler RNG init
        })
        .add_systems(Startup, setup_camera)
        .add_systems(Startup, setup_scene)
        .add_systems(Update, ray_cast) // the fixed system ↓
        .run();
}

fn setup_camera(mut commands: Commands, assets: Res<AssetServer>, mut rng: ResMut<RngResource>) {
    // let hdri_files: Vec<_> = std::fs::read_dir("assets/hdris")
    //     .unwrap()
    //     .filter_map(|entry| {
    //         let path = entry.unwrap().path();
    //         if path.is_file() {
    //             path.strip_prefix("assets")
    //                 .ok()
    //                 .and_then(|p| p.to_str())
    //                 .map(|s| s.to_string())
    //         } else {
    //             None
    //         }
    //     })
    //     .collect();

    // let random_hdri = hdri_files[rng.rng.random_range(0..hdri_files.len())].clone();
    let sky_handle = assets.load("hdris/im2.png");

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-3.0, 3.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        camera_controller::CameraController::default(),
        Skybox {
            image: sky_handle.clone(),
            brightness: 1000.0, // cd/m² scale for LDR monitors
            ..default()
        },
    ));
    commands.spawn(DirectionalLight {
        color: Color::WHITE,
        illuminance: 10000.0,
        ..default()
    });

    // ambient light
    // NOTE: The ambient light is used to scale how bright the environment map is so with a bright
    // environment map, use an appropriate color and brightness to match
    commands.insert_resource(AmbientLight {
        color: Color::srgb_u8(210, 220, 240),
        brightness: 1.0,
        ..default()
    });
    // commands.spawn(EnvironmentMapLight {
    //     diffuse_map: sky_handle.clone(),
    //     specular_map: sky_handle,
    //     intensity: 30_000.0,      // lux
    //     rotation: Quat::IDENTITY, // spin if you need sunrise
    //     ..default()
    // });
}

fn setup_scene(
    mut commands: Commands,
    mut rng: ResMut<RngResource>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    assets: Res<AssetServer>,
) {
    let scene = scene_generation::generate_scene(meshes, materials, rng, assets);
    for (mesh, material, transform) in scene {
        commands.spawn((Mesh3d(mesh), MeshMaterial3d(material), transform));
    }
}

fn ray_cast(rapier_context: ReadRapierContext) {
    // ReadRapierContext is a *wrapper*; grab the actual RapierContext:
    let ctx = rapier_context.single(); // ★ the missing line ★

    let ray_origin = Vec3::ZERO;
    let ray_direction = Vec3::new(0.0, 0.0, -1.0).normalize_or_zero(); // must be unit-length
    let max_distance = 100.0;
    let solid = true;
    let filter = QueryFilter::default();

    if let Some((entity, hit)) =
        ctx.cast_ray_and_get_normal(ray_origin, ray_direction, max_distance, solid, filter)
    {
        println!(
            "Ray hit {:?} at {:.2}\n  point  : {}\n  normal : {}",
            entity, hit.time_of_impact, hit.point, hit.normal
        );
    }
}
