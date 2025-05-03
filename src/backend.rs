use crate::depthmap; 
use smooth_bevy_cameras::controllers::unreal::{UnrealCameraBundle, UnrealCameraController};

use bevy::{
    core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass, NormalPrepass},
    pbr::NotShadowCaster,
    prelude::*,
};

pub fn setup(
    asset_server: Res<AssetServer>,
    mut depth_materials: ResMut<Assets<depthmap::PrepassOutputMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<depthmap::CustomMaterial>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands)
{
    // commands.spawn(SceneRoot(asset_server.load_with_settings(
    //     "odm_ultra/odm_textured_model_geo.obj",
    //     |settings: &mut bevy_obj::ObjSettings| {
    //         settings.force_compute_normals = true;
    //         settings.prefer_flat_normals = true;
    //     },
    // )));
    
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
        // // Disabling MSAA for maximum compatibility. Shader prepass with MSAA needs GPU capability MULTISAMPLED_SHADING
        // Msaa::Off,
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

    // A quad that shows the outputs of the prepass
    // To make it easy, we just draw a big quad right in front of the camera.
    // For a real application, this isn't ideal.
    commands.spawn((
        Mesh3d(meshes.add(Rectangle::new(20.0, 20.0))),
        MeshMaterial3d(depth_materials.add(depthmap::PrepassOutputMaterial {
            settings: depthmap::ShowPrepassSettings::default(),
        })),
        Transform::from_xyz(-0.75, 1.25, 3.0).looking_at(Vec3::new(2.0, -2.5, -5.0), Vec3::Y),
        NotShadowCaster,
    ));

    // light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(14.0, 18.0, 14.0),
    ));

    commands.spawn(Text::default()).with_child((
        TextSpan::new("Prepass Output: transparent\n"),
        // TextSpan::new("\n\n".to_string())
        // parent.spawn(TextSpan {
        //     value: "Controls\n".to_string(),
        //     style: style.clone(),
        // });
        // parent.spawn(TextSpan {
        //     value: "---------------\n".to_string(),
        //     style: style.clone(),
        // });
        // parent.spawn(TextSpan {
        //     value: "Space - Change output\n".to_string(),
        //     style,
        // });
    ));
    // commands.spawn(
    //     TextBundle::from_sections(vec![
    //         TextSection::new("Prepass Output: transparent\n", style.clone()),
    //         TextSection::new("\n\n", style.clone()),
    //         TextSection::new("Controls\n", style.clone()),
    //         TextSection::new("---------------\n", style.clone()),
    //         TextSection::new("Space - Change output\n", style),
    //     ])
    //     .with_style(Style {
    //         position_type: PositionType::Absolute,
    //         top: Val::Px(10.0),
    //         left: Val::Px(10.0),
    //         ..default()
    //     }),
    // );
}
