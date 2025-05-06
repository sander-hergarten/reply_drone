use std::{collections::HashMap, ops::Index};

use crossbeam_channel::Receiver;

use crate::{commands::{self, EngineResponses}, constants::{SCREENSHOT_HEIGHT, SCREENSHOT_WIDTH}, depthmap::{self, PrepassOutputMaterial}, input, screenshot::change_view, types::{FullGraphState, Position, Rotation, SingleNodeState}}; 
use crate::components::{ComponentIndex, DepthQuad};

use smooth_bevy_cameras::controllers::unreal::{UnrealCameraBundle, UnrealCameraController};

use bevy::{
    asset::RenderAssetUsages, core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass, NormalPrepass}, image::ImageSampler, math::quat, pbr::NotShadowCaster, prelude::*, render::{camera::{CameraOutputMode, RenderTarget}, render_resource::{TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor}}
};

// Resource for holding next camera and depth quad index
#[derive(Resource, Default)]
pub struct NextIndex(u32);

// A Bevy resource to hold the receiver end of the command channel
#[derive(Resource)]
pub struct CommandReceiver(Receiver<commands::EngineCommands>);

#[derive(Resource)]
pub struct RenderTargetImageHandle(Handle<Image>);

pub fn command_processing_system(
    mut commands: Commands,
    cmd_receiver: Res<CommandReceiver>, // Access the receiver resource
    mut next_index: ResMut<NextIndex>,
    mut query: Query<(&ComponentIndex, &mut Camera, &Transform), With<Camera3d>>,
    // needed for depth quad creation
    mut depth_materials: ResMut<Assets<depthmap::PrepassOutputMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    // needed for screenshot utility
    render_target_handle: Res<RenderTargetImageHandle>,
    images: Res<Assets<Image>>,
    // needed for custom shader
    writer: TextUiWriter,
    materials: ResMut<Assets<PrepassOutputMaterial>>,
    material_handle: Single<&MeshMaterial3d<PrepassOutputMaterial>>,
    text_entity: Single<Entity, With<Text>>,
) {
    change_view(
        materials,
        material_handle,
        1, // Force depth view
        text_entity,
        writer,
    );

    // Process all available commands in the channel
    for command in cmd_receiver.0.try_iter() {
        info!("Received command: {:?}", command);

        match command {
            commands::EngineCommands::AddNode { 
                position, 
                rotation,
                response_tx
            } => {
                let current_index = next_index.0;
                next_index.0 += 1;

                let direction: Quat = Quat::from_euler(
                    EulerRot::XYZ,
                    rotation[0] as f32,
                    rotation[1] as f32,
                    rotation[2] as f32
                );

                commands.spawn((
                    Camera3d::default(),
                    // This gives the camera a unique id
                    ComponentIndex(current_index),
                    Transform { 
                        translation: Vec3::new(
                            position[0] as f32,
                            position[1] as f32,
                            position[2] as f32
                        ), 
                        rotation: direction,
                        scale: Vec3::ONE
                    },
                    // Disabling MSAA for maximum compatibility. Shader prepass with MSAA needs GPU capability MULTISAMPLED_SHADING
                    Msaa::Off,
                    // To enable the prepass you need to add the components associated with the ones you need
                    // This will write the depth buffer to a texture that you can use in the main pass
                    DepthPrepass,
                    // This will generate a texture containing world normals (with normal maps applied)
                    NormalPrepass,
                    // This will generate a texture containing screen space pixel motion vectors
                    MotionVectorPrepass,
                ));

                commands.spawn((
                    Mesh3d(meshes.add(Rectangle::new(20.0, 20.0))),
                    // quad has same index as camera
                    ComponentIndex(current_index),
                    // also has a unique marker to differentiate
                    DepthQuad,
                    NotShadowCaster,
                    MeshMaterial3d(depth_materials.add(depthmap::PrepassOutputMaterial {
                        settings: depthmap::ShowPrepassSettings::default(),
                    })),
                    Transform { 
                        translation: Vec3::new(
                            position[0] as f32,
                            position[1] as f32,
                            position[2] as f32
                        ) 
                        + direction * Vec3::NEG_Z // (-Z is Bevy's forward)
                        + direction * Vec3::Y // (+Y is Bevy's up)
                        + direction * Vec3::X, // (+X is Bevy's right)
                        rotation: direction,
                        scale: Vec3::ONE
                    },
                ));

                info!("Spawned camera entity with corresponding depth quad (ID: {})", current_index);

                // return id if response token was given
                if let Some(tx) = response_tx {
                    if let Err(e) = tx.send(EngineResponses::AddNodeResponse { 
                        id: current_index 
                    }) {
                        error!("Failed to send NodeAdded response: {}", e);
                    }
                }
            },
            commands::EngineCommands::GetFullGraphState {
                response_tx
            } => {
                if let Some(tx) = response_tx {
                    let mut full_graph_state: FullGraphState = HashMap::new();

                    for (index, mut camera, transform) in query.iter_mut() {
                        camera.target = RenderTarget::Image(render_target_handle.0.clone());

                        if let Some(rendered_image) = images.get(&render_target_handle.0) {
                            let position: Position = [
                                transform.translation.x as i32,
                                transform.translation.y as i32,
                                transform.translation.z as i32,
                            ];
                            let euler_rot = transform.rotation.to_euler(EulerRot::XYZ);
                            let rotation: Rotation = [
                                euler_rot.0 as i32,
                                euler_rot.1 as i32,
                                euler_rot.2 as i32
                            ];

                            full_graph_state.insert(index.0, SingleNodeState {
                                id: index.0,
                                position,
                                rotation,
                                depth_map: rendered_image.clone(),
                                postion_of_other_nodes: Vec::new(),
                                rotation_of_other_nodes: Vec::new(),
                            });
                        }
                    }

                    if let Err(e) = tx.send(EngineResponses::FullGraphStateResponse { 
                        full_graph_state
                    }) {
                        error!("Failed to send NodeAdded response: {}", e);
                    }
                }
            }
        }
    }
}

pub fn setup(
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    // create image handler
    let render_target_image = Image::new( 
        bevy::render::render_resource::Extent3d {
            width: SCREENSHOT_WIDTH,
            height: SCREENSHOT_HEIGHT,
            depth_or_array_layers: 1,
        },
        bevy::render::render_resource::TextureDimension::D2,
        Vec::new(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::all(),
    );
    let render_target_handle = images.add(render_target_image);
    commands.insert_resource(RenderTargetImageHandle(render_target_handle.clone()));

    commands.spawn(
        SceneRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("centered.glb")))
    );

    // Light 
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(-40.0, -45.0, -6.0)
    ));

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.0, 3., 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        // Disabling MSAA for maximum compatibility. Shader prepass with MSAA needs GPU capability MULTISAMPLED_SHADING
        Msaa::Off,
        // To enable the prepass you need to add the components associated with the ones you need
        // This will write the depth buffer to a texture that you can use in the main pass
        DepthPrepass,
        // This will generate a texture containing world normals (with normal maps applied)
        NormalPrepass,
        // This will generate a texture containing screen space pixel motion vectors
        MotionVectorPrepass,
    )).insert(UnrealCameraBundle::new(
            UnrealCameraController::default(),
            Vec3::new(-2.0, 5.0, 5.0),
            Vec3::new(0., 0., 0.),
            Vec3::Y,
        ));



    commands.spawn(Text::default()).with_child((
        TextSpan::new("Prepass Output: transparent\n"),
    ));
}
