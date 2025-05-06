use bevy::math::Vec3;

pub const PREPASS_SHADER_ASSET_PATH: &str = "shaders/show_prepass.wgsl";
pub const MATERIAL_SHADER_ASSET_PATH: &str = "shaders/custom_material.wgsl";

pub const SCREENSHOT_WIDTH: u32 = 1920;
pub const SCREENSHOT_HEIGHT: u32 = 1080;

const SCREENSHOT_CAMERA_POS: Vec3 = Vec3::new(0.0, 5.0, 15.0);
const SCREENSHOT_LOOK_AT: Vec3 = Vec3::new(2.0, -2.5, -5.0);
