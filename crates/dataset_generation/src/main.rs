//! Load a cubemap texture onto a cube like a skybox and cycle through different compressed texture formats
mod components;
mod features;
mod image_capture;
mod scene_generation;

use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::{
    core_pipeline::Skybox,
    prelude::*,
    render::{
        camera::{CameraRenderGraph, Viewport},
        render_resource::{TextureViewDescriptor, TextureViewDimension},
        view::RenderLayers,
    },
};
use bevy_rapier3d::prelude::*;
use components::*;
use features::*;
use image_capture::*;
use iyes_perf_ui::prelude::*;
use rand::prelude::*;
use smooth_bevy_cameras::{
    LookTransformPlugin,
    controllers::unreal::{UnrealCameraBundle, UnrealCameraController, UnrealCameraPlugin},
};

#[derive(Resource)]
struct RngResource {
    rng: SmallRng,
}

impl RngResource {
    fn sample_normal(&mut self, mean: f32, std_dev: f32) -> f32 {
        // Box-Muller transform
        let u1 = self.rng.random::<f32>();
        let u2 = self.rng.random::<f32>();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        mean + std_dev * z0
    }
}

#[derive(Resource)]
struct Cubemap {
    currently_loaded: Handle<Image>,
    all_handles: Vec<Handle<Image>>,
}

impl Cubemap {
    fn new() -> Self {
        Self {
            currently_loaded: Handle::<Image>::default(),
            all_handles: Vec::new(),
        }
    }
    fn fill(&mut self, assets: &Res<AssetServer>) {
        for entry in std::fs::read_dir("assets/hdr").expect("hdrs missing") {
            let path = entry.unwrap().path();
            let name = path.strip_prefix("assets").unwrap();
            self.all_handles.push(assets.load(name));
        }
    }
    fn pick_random(&mut self, rng: &mut RngResource) -> Handle<Image> {
        self.currently_loaded =
            self.all_handles[rng.rng.random_range(0..self.all_handles.len())].clone();
        self.currently_loaded.clone()
    }
}

#[derive(Resource)]
struct SpawnRange {
    min: Vec3,
    max: Vec3,
}

#[derive(Resource)]
struct FeaturesSpawned(u16);

#[derive(Resource)]
struct Textures(Vec<Handle<Image>>, Vec<String>);

#[derive(Event)]
struct CameraChangeEvent(Transform);

#[derive(Event)]
struct RegenerateSceneEvent;

#[derive(Event)]
struct GenerateSceneEvent;

#[derive(Event)]
struct ChangeEnvironmentEvent;

impl Textures {
    fn new() -> Self {
        Self(Vec::new(), Vec::new())
    }
    fn fill(&mut self, assets: &Res<AssetServer>) {
        let mut names = Vec::new();
        for entry in std::fs::read_dir("assets/textures").expect("textures folder missing") {
            let path = entry.unwrap().path();
            let name = path.strip_prefix("assets").unwrap();
            names.push(name.to_str().unwrap().to_string());
            self.0.push(assets.load(name));
        }
        self.1 = names;
    }

    fn pick_random(&self, rng: &mut RngResource) -> &str {
        &self.1[rng.rng.random_range(0..self.1.len())]
    }
}

#[derive(Resource)]
struct FeatureTextures(Vec<Handle<Image>>);

impl FeatureTextures {
    fn new() -> Self {
        Self(Vec::new())
    }
    fn fill(&mut self, assets: &Res<AssetServer>) {
        for entry in std::fs::read_dir("assets/features").expect("features folder missing") {
            self.0
                .push(assets.load(entry.unwrap().path().strip_prefix("assets").unwrap()));
        }
    }
}

#[derive(Event)]
struct ClearSceneEvent;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins, // Make sure pipelines are ready before rendering
                            // .set(RenderPlugin {
                            //     synchronous_pipeline_compilation: true,
                            //     ..default()
                            // }),
        )
        .add_plugins(LookTransformPlugin)
        .add_plugins(UnrealCameraPlugin::default())
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(FrameTimeDiagnosticsPlugin)
        .add_plugins(PerfUiPlugin)
        // .add_plugins(ScheduleRunnerPlugin {
        //     run_mode: RunMode::Loop { wait: None },
        // })
        // .add_plugins(bevy_capture::CapturePlugin)
        .insert_resource(Textures::new())
        .insert_resource(FeatureTextures::new())
        .insert_resource(FeaturesSpawned(0))
        .insert_resource(RngResource {
            rng: SmallRng::from_rng(&mut rand::rng()), // <- simpler RNG init
        })
        .insert_resource(SpawnRange {
            min: Vec3::new(-10.0, -10.0, -10.0),
            max: Vec3::new(10.0, 10.0, 10.0),
        })
        .insert_resource(Cubemap::new())
        .add_event::<CameraChangeEvent>()
        .add_event::<GenerateSceneEvent>()
        .add_event::<ClearSceneEvent>()
        .add_event::<ChangeEnvironmentEvent>()
        .add_event::<CaptureImageEvent>()
        .add_event::<RegenerateSceneEvent>()
        .add_systems(PreStartup, setup_assets)
        .add_systems(Startup, setup)
        .add_systems(Update, sync_cameras)
        .add_systems(Update, environment_loader.after(setup))
        // .add_systems(Update, transform_camera)
        // .add_systems(Update, after_render)
        // .add_systems(Last, clear_objects)
        .add_systems(Update, emit_regenerate_scene_event_on_button_press)
        .add_systems(
            Update,
            clear_scene.after(emit_regenerate_scene_event_on_button_press),
        )
        .add_systems(
            Update,
            generate_scene
                .after(clear_scene)
                .after(emit_regenerate_scene_event_on_button_press),
        )
        .add_systems(Update, spawn_features)
        .add_systems(
            Update,
            change_environment.after(emit_regenerate_scene_event_on_button_press),
        )
        .add_systems(Update, regenerate_on_space_key)
        .add_systems(Update, swap_and_capture.run_if(is_headless))
        .run();
}

fn after_render(
    mut camera_change_event: EventWriter<CameraChangeEvent>,
    mut rng: ResMut<RngResource>,
    spawn_range: Res<SpawnRange>,
) {
    let position = Vec3::new(
        rng.rng.random_range(spawn_range.min.x..spawn_range.max.x),
        rng.rng.random_range(spawn_range.min.y..spawn_range.max.y),
        rng.rng.random_range(spawn_range.min.z..spawn_range.max.z),
    );

    let random_rotation = Quat::from_euler(
        EulerRot::XYZ,
        rng.rng.random_range(0.0..2.0 * std::f32::consts::PI),
        rng.rng.random_range(0.0..2.0 * std::f32::consts::PI),
        rng.rng.random_range(0.0..2.0 * std::f32::consts::PI),
    );
    camera_change_event.send(CameraChangeEvent(Transform {
        translation: position,
        rotation: random_rotation,
        ..default()
    }));
}

fn transform_camera(
    mut cameras: Query<&mut Transform, With<Camera>>,
    mut camera_change_event: EventReader<CameraChangeEvent>,
) {
    for mut transform in &mut cameras {
        for event in camera_change_event.read() {
            transform.translation = event.0.translation;
            transform.rotation = event.0.rotation;
        }
    }
}

fn setup_assets(
    mut textures: ResMut<Textures>,
    mut feature_textures: ResMut<FeatureTextures>,
    asset_server: Res<AssetServer>,
) {
    textures.fill(&asset_server);
    feature_textures.fill(&asset_server);
    println!("Assets loaded");
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    windows: Query<&Window>,
    mut generate_scene_event_writer: EventWriter<GenerateSceneEvent>,
    mut skyboxes: Query<&mut Skybox>,
    mut cubemap: ResMut<Cubemap>,
    mut rng: ResMut<RngResource>,
    mut images: ResMut<Assets<Image>>,
) {
    cubemap.fill(&asset_server);

    let skybox_handle = cubemap.pick_random(&mut rng);

    #[cfg(not(feature = "headless"))]
    {
        let window = windows.single();
        let width = window.resolution.physical_width();
        let height = window.resolution.physical_height();
        let half_width = width / 2;
        commands.spawn((
            Camera {
                order: 1, // Set the main camera to render first
                viewport: Some(Viewport {
                    physical_position: UVec2::new(0, 0),
                    physical_size: UVec2::new(half_width, height),
                    ..default()
                }),
                ..default()
            },
            Camera3d::default(),
            UnrealCameraBundle::new(
                UnrealCameraController::default(),
                Vec3::new(-2.0, 5.0, 5.0),
                Vec3::new(0., 0., 0.),
                Vec3::Y,
            ),
            Skybox {
                image: skybox_handle.clone(),
                brightness: 1000.0,
                ..default()
            },
            EnvironmentMapLight {
                diffuse_map: skybox_handle.clone(),
                specular_map: skybox_handle.clone(),
                intensity: 1000.0,        // lux
                rotation: Quat::IDENTITY, // spin if you need sunrise
            },
        ));

        commands.spawn((
            Camera3d::default(),
            Camera {
                viewport: Some(Viewport {
                    physical_position: UVec2::new(half_width, 0),
                    physical_size: UVec2::new(half_width, height),
                    ..default()
                }),
                order: 0,
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            },
            CameraRenderGraph::new(bevy::core_pipeline::core_3d::graph::Core3d),
            RenderLayers::layer(1),
        ));
    }

    // cameras saving
    #[cfg(feature = "headless")]
    {
        let default_target = new_target(&mut images);
        commands.spawn((
            Camera {
                order: 0, // Set the main camera to render first
                target: RenderTarget::Image(default_target.clone().into()),
                ..default()
            },
            Camera3d::default(),
            Skybox {
                image: skybox_handle.clone(),
                brightness: 1000.0,
                ..default()
            },
            EnvironmentMapLight {
                diffuse_map: skybox_handle.clone(),
                specular_map: skybox_handle.clone(),
                intensity: 1000.0,        // lux
                rotation: Quat::IDENTITY, // spin if you need sunrise
            },
            CaptureBundle::default(),
        ));

        let selection_target = new_target(&mut images);
        commands.spawn((
            Camera3d::default(),
            Camera {
                order: 1,
                target: RenderTarget::Image(selection_target.clone().into()),
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            },
            CameraRenderGraph::new(bevy::core_pipeline::core_3d::graph::Core3d),
            RenderLayers::layer(1),
            CaptureBundle::default(),
        ));

        commands.insert_resource(CaptureState::new(default_target, selection_target));
    }

    commands.insert_resource(AmbientLight {
        color: Color::srgb_u8(210, 220, 240),
        brightness: 0.0,
    });

    #[cfg(not(feature = "headless"))]
    commands.spawn((
        PerfUiRoot {
            display_labels: false,
            layout_horizontal: true,
            values_col_width: 32.0,
            ..default()
        },
        PerfUiEntryFPSWorst::default(),
        PerfUiEntryFPS::default(),
    ));
}

fn generate_scene(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut textures: ResMut<Textures>,
    rng: ResMut<RngResource>,
    meshes: ResMut<Assets<Mesh>>,
    assets: Res<AssetServer>,
    spawn_range: Res<SpawnRange>,
    mut generate_scene_event: EventReader<GenerateSceneEvent>,
    mut features_spawned: ResMut<FeaturesSpawned>,
) {
    if generate_scene_event.is_empty() {
        return;
    } else {
        generate_scene_event.clear();
    }

    let black = materials.add(StandardMaterial {
        base_color: Color::BLACK,
        unlit: true,
        ..default()
    });

    let scene =
        scene_generation::generate_scene(meshes, materials, rng, spawn_range, textures, assets);

    let textured_objects =
        scene
            .clone()
            .into_iter()
            .map(|(mesh, material, transform, collider)| {
                (
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(material.clone()),
                    transform.clone(),
                    collider.clone(),
                    IsCuboid,
                )
            });

    let blacks = (0..scene.len()).map(|_| black.clone()).collect::<Vec<_>>();
    let non_textured_objects =
        scene
            .into_iter()
            .zip(blacks.into_iter())
            .map(|((mesh, _, transform, _), black)| {
                (
                    Mesh3d(mesh),
                    MeshMaterial3d(black),
                    transform,
                    IsCuboid,
                    RenderLayers::layer(1),
                )
            });

    println!("textured_objects: {}", textured_objects.len());
    commands.spawn_batch(textured_objects);
    commands.spawn_batch(non_textured_objects);
    features_spawned.0 = 0;
}

fn environment_loader(
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut cubemap: ResMut<Cubemap>,
    mut skyboxes: Query<&mut Skybox>,
    mut environment_map_light: Query<&mut EnvironmentMapLight>,
) {
    if asset_server
        .load_state(&cubemap.currently_loaded)
        .is_loaded()
    {
        let image = images.get_mut(&cubemap.currently_loaded).unwrap();
        // NOTE: PNGs do not have any metadata that could indicate they contain a cubemap texture,
        // so they appear as one texture. The following code reconfigures the texture as necessary.
        if image.texture_descriptor.array_layer_count() == 1 {
            image.reinterpret_stacked_2d_as_array(image.height() / image.width());
            image.texture_view_descriptor = Some(TextureViewDescriptor {
                dimension: Some(TextureViewDimension::Cube),
                ..default()
            });
        }

        for mut skybox in &mut skyboxes {
            skybox.image = cubemap.currently_loaded.clone();
        }

        for mut environment_map_light in &mut environment_map_light {
            environment_map_light.diffuse_map = cubemap.currently_loaded.clone();
            environment_map_light.specular_map = cubemap.currently_loaded.clone();
        }
    }
}

fn change_environment(
    mut cubemap: ResMut<Cubemap>,
    mut skyboxes: Query<&mut Skybox>,
    mut environment_map_light: Query<&mut EnvironmentMapLight>,
    mut rng: ResMut<RngResource>,
    mut change_environment_event: EventReader<ChangeEnvironmentEvent>,
    mut images: ResMut<Assets<Image>>,
) {
    if change_environment_event.is_empty() {
        return;
    } else {
        change_environment_event.clear();
    }

    let new_cubemap = cubemap.pick_random(&mut rng);
    println!("new_cubemap: {:?}", new_cubemap);

    let image = images.get_mut(&new_cubemap).unwrap();

    if image.texture_descriptor.array_layer_count() == 1 {
        image.reinterpret_stacked_2d_as_array(image.height() / image.width());
        image.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..default()
        });
    }
    for mut skybox in &mut skyboxes {
        skybox.image = new_cubemap.clone();
    }

    for mut environment_map_light in &mut environment_map_light {
        environment_map_light.diffuse_map = new_cubemap.clone();
        environment_map_light.specular_map = new_cubemap.clone();
        // Sample from normal distribution and ensure non-negative value
        let intensity = rng.sample_normal(1000.0, 800.0).max(40.0);
        println!("intensity: {}", intensity);
        environment_map_light.intensity = intensity;
    }
}

fn sync_cameras(
    main_camera: Query<&Transform, (With<Camera>, Without<RenderLayers>)>,
    mut second_camera: Query<&mut Transform, (With<Camera>, With<RenderLayers>)>,
) {
    if let (Ok(main_transform), Ok(mut second_transform)) =
        (main_camera.get_single(), second_camera.get_single_mut())
    {
        *second_transform = *main_transform;
    }
}

// fn save_image(
//     mut images: ResMut<Assets<Image>>,
//     mut cubemap: ResMut<Cubemap>,
//     mut skyboxes: Query<&mut Skybox>,
// ) {
//     image.copy_from_slice(cubemap.image_handle.as_bytes());
//     images.add(image);
// }

fn clear_scene(
    mut commands: Commands,
    entities: Query<Entity, With<Mesh3d>>,
    mut clear_scene_event: EventReader<ClearSceneEvent>,
) {
    if !clear_scene_event.is_empty() {
        clear_scene_event.clear();
        for entity in entities.iter() {
            commands.entity(entity).despawn_recursive();
        }
    }
}

fn emit_regenerate_scene_event_on_button_press(
    mut clear_scene_event_writer: EventWriter<ClearSceneEvent>,
    mut generate_scene_event_writer: EventWriter<GenerateSceneEvent>,
    mut change_environment_event_writer: EventWriter<ChangeEnvironmentEvent>,
    mut regenerate_scene_event_writer: EventReader<RegenerateSceneEvent>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
) {
    if regenerate_scene_event_writer.is_empty() {
        return;
    } else {
        regenerate_scene_event_writer.clear();
    }

    clear_scene_event_writer.send(ClearSceneEvent);
    generate_scene_event_writer.send(GenerateSceneEvent);
    change_environment_event_writer.send(ChangeEnvironmentEvent);
}

fn is_headless() -> bool {
    cfg!(feature = "headless")
}

fn regenerate_on_space_key(
    mut regenerate_scene_event_writer: EventWriter<RegenerateSceneEvent>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
) {
    if keyboard_input.just_pressed(KeyCode::Space) {
        regenerate_scene_event_writer.send(RegenerateSceneEvent);
    }
}
