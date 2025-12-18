use bevy::{
    asset::Assets,
    color::Color,
    ecs::{
        entity::Entity,
        query::With,
        system::{ResMut, Single},
    },
    log,
    pbr::MeshMaterial3d,
    ui::widget::{Text, TextUiWriter},
};

use crate::depthmap::PrepassOutputMaterial;

pub fn change_view(
    mut materials: ResMut<Assets<PrepassOutputMaterial>>,
    material_handle: Single<&MeshMaterial3d<PrepassOutputMaterial>>,
    view_index: u32,
    text_entity: Single<Entity, With<Text>>,
    mut writer: TextUiWriter,
) {
    let label = match view_index {
        0 => "transparent",
        1 => "depth",
        2 => "normals",
        3 => "motion vectors",
        _ => unreachable!(),
    };
    let text_entity = *text_entity;
    *writer.text(text_entity, 1) = format!("Prepass Output: {label}\n");
    writer.for_each_color(text_entity, |mut color| {
        color.0 = Color::WHITE;
    });

    if let Some(mat) = materials.get_mut(*material_handle) {
        mat.settings.show_depth = (view_index == 1) as u32;
        mat.settings.show_normals = (view_index == 2) as u32;
        mat.settings.show_motion_vectors = (view_index == 3) as u32;
    } else {
        log::error!("Failed to get PrepassOutputMaterial to change view");
    }
}

// pub fn take_depth_screenshot (
//     mut commands: Commands,
//     materials: ResMut<Assets<PrepassOutputMaterial>>,
//     material_handle: Single<&MeshMaterial3d<PrepassOutputMaterial>>,
//     text_entity: Single<Entity, With<Text>>,
//     writer: TextUiWriter,
//     mut camera_query: Query<
//         (&ComponentIndex, &Transform),
//         (With<Camera3d>, Without<DepthQuad>)
//     >,
//     mut quad_query: Query<
//         (&ComponentIndex, &Transform),
//         (With<DepthQuad>, Without<Camera3d>)
//     >,
//     idx: u32,
// ) {
//     log::info!("Preparing for depth screenshot...");
//
//     // 1. Set the view to Depth (view_index = 1)
//     change_view(
//         materials,
//         material_handle,
//         1, // Force depth view
//         text_entity,
//         writer,
//     );

// for (camera_index, camera_transform) in camera_query.iter_mut() {
//     if camera_index.0 == idx {
// // Move the camera to the desired position and make it look at the target
// camera_transform.translation = SCREENSHOT_CAMERA_POS;
// camera_transform.look_at(SCREENSHOT_LOOK_AT, Vec3::Y);

// Now, position the quad relative to the camera's position and orientation
// for (quad_index, quad) in quad_query.iter_mut() {
//     if (quad_index.0 == idx) {
// Get the camera's local axes
// let camera_forward = camera_transform.forward();
// let camera_right = camera_transform.right();
// let camera_up = camera_transform.up();
// // Calculate the quad's position by adding the (1, 1, 1) offset along the camera's local axes
// quad.translation = camera_transform.translation +
//     camera_forward * 1.0 + // 1 unit forward from the camera
//     camera_right * 1.0 +   // 1 unit to the right of the camera
//     camera_up * 1.0;     // 1 unit up from the camera

// TODO change direction
// Make the quad look at the same point the camera is looking at
// This ensures the quad is facing in a similar direction as the camera
// quad.look_at(SCREENSHOT_LOOK_AT, Vec3::Y);
//             }
//         }
//     }
// }
//

// 3. Take the screenshot using the new API and return the screenshot data
// commands.spawn(
//     Screenshot::primary_window().observe(|output: ScreenshotOutput| {
//         match output {
//             ScreenshotOutput::Image(image) => {
//                 let width = image.size().x as usize;
//                 let height = image.size().y as usize;
//                 let data = image.data();
//
//                 Some(Array3::from_shape_vec((height, width, 4), data.clone()).unwrap())
//             }
//             ScreenshotOutput::Error(err) => {
//                 eprintln!("Failed to capture screenshot: {:?}", err);
//             }
//         }
//     }),
// );

// log::info!("Screenshot requested from camera ID: {}", idx);
// Note: Saving happens asynchronously in the background by Bevy's render systems.
// There's no immediate confirmation here, but the observer handles it.
// }
