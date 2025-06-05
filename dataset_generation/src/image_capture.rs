use crate::RegenerateSceneEvent;
use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        camera::RenderTarget,
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
        view::{
            RenderLayers,
            screenshot::{Screenshot, save_to_disk},
        },
    },
};
use bevy_capture::{
    CameraTargetHeadless, Capture, CaptureBundle,
    encoder::frames::{self, FramesEncoder},
};
#[derive(Event)]
pub struct CaptureImageEvent;

#[derive(Resource)]
pub struct CaptureState {
    frame: u32,
    current_default_target: Handle<Image>,
    current_selection_target: Handle<Image>,
}

impl CaptureState {
    pub fn new(
        current_default_target: Handle<Image>,
        current_selection_target: Handle<Image>,
    ) -> Self {
        Self {
            frame: 0,
            current_default_target,
            current_selection_target,
        }
    }
}

pub fn new_target(images: &mut Assets<Image>) -> Handle<Image> {
    const WIDTH: u32 = 1024;
    const HEIGHT: u32 = 1024;
    let size = Extent3d {
        width: WIDTH,
        height: HEIGHT,
        depth_or_array_layers: 1,
    };
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0; WIDTH as usize * HEIGHT as usize * 4],
        TextureFormat::Bgra8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage =
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING;
    images.add(image)
}

pub fn swap_and_capture(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut state: ResMut<CaptureState>,
    mut default_camera: Query<&mut Camera, Without<RenderLayers>>,
    mut selection_camera: Query<&mut Camera, With<RenderLayers>>,
    mut capture_image_event: EventReader<CaptureImageEvent>,
    mut regenerate_scene_event: EventWriter<RegenerateSceneEvent>,
) {
    if capture_image_event.is_empty() {
        return;
    }

    capture_image_event.clear();

    // 1. save the image that *just finished* rendering
    let default_path = format!(
        "/Users/sanderhergarten/datasources/reply_drone_features/frames/default/frame_{}.png",
        state.frame
    );
    let selection_path = format!(
        "/Users/sanderhergarten/datasources/reply_drone_features/frames/selection/frame_{}.png",
        state.frame
    );
    commands
        .spawn(Screenshot(RenderTarget::Image(
            state.current_default_target.clone().into(),
        )))
        .observe(save_to_disk(default_path));

    commands
        .spawn(Screenshot(RenderTarget::Image(
            state.current_selection_target.clone().into(),
        )))
        .observe(save_to_disk(selection_path));

    // 2. create a fresh target for the *next* frame
    let new_default_target = new_target(&mut images);
    let new_selection_target = new_target(&mut images);

    // 3. point the camera at it
    let mut default_cam = default_camera.single_mut();
    default_cam.target = RenderTarget::Image(new_default_target.clone().into());

    let mut selection_cam = selection_camera.single_mut();
    selection_cam.target = RenderTarget::Image(new_selection_target.clone().into());

    // 4. remember it for the next loop
    state.current_default_target = new_default_target;
    state.current_selection_target = new_selection_target;
    state.frame += 1;
    regenerate_scene_event.send(RegenerateSceneEvent);
}

// pub fn swap_and_capture(
//     mut commands: Commands,
//     mut images: ResMut<Assets<Image>>,
//     mut state: ResMut<CaptureState>,
//     mut default_camera: Query<&mut Capture, Without<RenderLayers>>,
//     mut selection_camera: Query<&mut Capture, With<RenderLayers>>,
//     mut capture_image_event: EventReader<CaptureImageEvent>,
//     mut regenerate_scene_event: EventWriter<RegenerateSceneEvent>,
// ) {
//     if capture_image_event.is_empty() {
//         return;
//     }

//     capture_image_event.clear();
//     let mut default_camera = default_camera.single_mut();
//     if !default_camera.is_capturing() {
//         default_camera.start(FramesEncoder::new(
//             "/Users/sanderhergarten/datasources/reply_drone_features/frames/default/",
//         ))
//     }
//     let mut selection_camera = selection_camera.single_mut();
//     if !selection_camera.is_capturing() {
//         selection_camera.start(FramesEncoder::new(
//             "/Users/sanderhergarten/datasources/reply_drone_features/frames/selection/",
//         ));
//     }

//     // // 3. point the camera at it
//     // let mut default_cam = default_camera.single_mut();
//     // default_cam.target = RenderTarget::Image(new_default_target.clone().into());

//     // let mut selection_cam = selection_camera.single_mut();
//     // selection_cam.target = RenderTarget::Image(new_selection_target.clone().into());

//     // 4. remember it for the next loop
//     // state.current_default_target = new_default_target;
//     // state.current_selection_target = new_selection_target;
//     state.frame += 1;
//     regenerate_scene_event.send(RegenerateSceneEvent);
// }
