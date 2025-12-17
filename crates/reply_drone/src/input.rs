// use bevy::{
//     asset::Assets,
//     color::Color,
//     core_pipeline::core_3d::Camera3d,
//     ecs::{
//         component::Component, entity::Entity, query::{With, Without}, system::{Commands, Local, Query, Res, ResMut, Single}
//     },
//     input::{keyboard::KeyCode, ButtonInput},
//     log,
//     math::{Quat, Vec3},
//     pbr::MeshMaterial3d,
//     render::view::screenshot::{save_to_disk, Screenshot},
//     transform::components::Transform,
//     ui::widget::{Text, TextUiWriter},
// };

// use crate::{backend::CameraIndex, depthmap::PrepassOutputMaterial};


// Every time you press space, it will cycle between transparent, depth and normals view
// pub fn toggle_prepass_view(
//     keycode: Res<ButtonInput<KeyCode>>,
//     material_handle: Single<&MeshMaterial3d<PrepassOutputMaterial>>,
//     materials: ResMut<Assets<PrepassOutputMaterial>>,
//     mut prepass_view: Local<u32>,
//     text: Single<Entity, With<Text>>,
//     writer: TextUiWriter,
// ) {
//     if keycode.just_pressed(KeyCode::Space) {
//         *prepass_view = (*prepass_view + 1) % 4;
//         change_view(materials, material_handle, *prepass_view, text, writer);
//     }
// }
//
// pub fn take_depth_screenshot_on_tab(
//     keycode: Res<ButtonInput<KeyCode>>,
// ) {
//     if keycode.just_pressed(KeyCode::Tab) {
//     }
// }
