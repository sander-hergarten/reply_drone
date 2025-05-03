use bevy::{
    asset::Assets,
    color::Color,
    core_pipeline::core_3d::Camera3d,
    ecs::{
        entity::Entity,
        // Added Commands for spawning the screenshot entity
        // Removed Resource (no longer needed for ScreenshotManager)
        system::{Commands, Local, Query, Res, ResMut, Single},
        query::With,
    },
    input::{keyboard::KeyCode, ButtonInput},
    log,
    math::Vec3,
    pbr::MeshMaterial3d,
    render::view::screenshot::{Screenshot, save_to_disk},
    transform::components::Transform,
    ui::widget::{Text, TextUiWriter},
};

use crate::depthmap::PrepassOutputMaterial;

// --- toggle_prepass_view remains the same ---
/// Every time you press space, it will cycle between transparent, depth and normals view
pub fn toggle_prepass_view(
    keycode: Res<ButtonInput<KeyCode>>,
    material_handle: Single<&MeshMaterial3d<PrepassOutputMaterial>>,
    materials: ResMut<Assets<PrepassOutputMaterial>>,
    mut prepass_view: Local<u32>,
    text: Single<Entity, With<Text>>,
    writer: TextUiWriter,
) {
    if keycode.just_pressed(KeyCode::Space) {
        *prepass_view = (*prepass_view + 1) % 4;
        change_view(materials, material_handle, *prepass_view, text, writer);
    }
}

// --- change_view remains the same ---
fn change_view(
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

// --- REVISED: System for taking depth screenshot ---

const SCREENSHOT_CAMERA_POS: Vec3 = Vec3::new(0.0, 5.0, 15.0);
const SCREENSHOT_LOOK_AT: Vec3 = Vec3::ZERO;

pub fn take_depth_screenshot_on_tab(
    keycode: Res<ButtonInput<KeyCode>>,
    // --- Resources needed to change view ---
    material_handle: Single<&MeshMaterial3d<PrepassOutputMaterial>>,
    materials: ResMut<Assets<PrepassOutputMaterial>>,
    text_entity: Single<Entity, With<Text>>,
    writer: TextUiWriter,
    // --- Resources needed for camera manipulation ---
    mut camera_query: Query<&mut Transform, With<Camera3d>>,
    // --- Commands needed for spawning screenshot entity ---
    mut commands: Commands, // Added Commands
    // --- ScreenshotManager and Window Query Removed ---
) {
    if keycode.just_pressed(KeyCode::Tab) {
        log::info!("Tab pressed - preparing for depth screenshot...");

        // 1. Set the view to Depth (view_index = 1)
        change_view(
            materials,
            material_handle,
            1, // Force depth view
            text_entity,
            writer,
        );

        // 2. Move the camera
        match camera_query.get_single_mut() {
            Ok(mut camera_transform) => {
                camera_transform.translation = SCREENSHOT_CAMERA_POS;
                camera_transform.look_at(SCREENSHOT_LOOK_AT, Vec3::Y);
                log::info!("Camera moved to snapshot position.");
            }
            Err(e) => {
                log::error!("Failed to get unique camera transform: {e}. Cannot take screenshot.");
                return;
            }
        }

        // 3. Take the screenshot using the new API
        let path = "./depth_screenshot.png"; // Save in current directory
        // Spawn an entity with the Screenshot component and attach the save_to_disk observer
        commands.spawn(Screenshot::primary_window())
            .observe(save_to_disk(path)); // Use the observer to save the file

        log::info!("Screenshot requested, will be saved to {}", path);
        // Note: Saving happens asynchronously in the background by Bevy's render systems.
        // There's no immediate confirmation here, but the observer handles it.
    }
}
