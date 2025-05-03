mod depthmap;
mod input;
mod backend;
mod reader;

mod helper;
use helper::Position;

use bevy::prelude::*;
use pyo3::prelude::*;

use smooth_bevy_cameras::{
    controllers::unreal::UnrealCameraPlugin,
    LookTransformPlugin,
};

pub type Rotation = [i32; 3];
pub type NodeId = u32;
pub type Seed = u64;

#[pyclass]
#[derive(Resource)]
pub struct Engine {
    pub node_positions: Vec<Position>,
    pub node_rotations: Vec<Rotation>,
}

#[derive(Resource)]
struct SharedData {
    engine: Engine,
}

fn run_bevy_app(engine: Engine) {
    println!("[Rust] Starting Bevy app...");
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(LookTransformPlugin)
        .add_plugins(UnrealCameraPlugin::default())
        .add_plugins((
            MaterialPlugin::<depthmap::CustomMaterial>::default(),
            MaterialPlugin::<depthmap::PrepassOutputMaterial> {
                // This material only needs to read the prepass textures,
                // but the meshes using it should not contribute to the prepass render, so we can disable it.
                prepass_enabled: false,
                ..default()
            }
        ))
        .add_systems(Startup, backend::setup)
        .add_systems(Update, (input::toggle_prepass_view, input::take_depth_screenshot_on_tab))
        .insert_resource(engine)
        .run();
    println!("[Rust] Bevy app finished.");
}

#[pyfunction]
fn start(node_amount: usize) -> PyResult<()> {
    let bounds = reader::read_glb_bounds("centered.glb");
    let mut node_positions = Vec::with_capacity(node_amount);
    let mut node_rotations = Vec::with_capacity(node_amount);
    for _ in 0..node_amount {
        let position: Position = helper::random_position(bounds);
        node_positions.push(position);
        node_rotations.push(helper::generate_camera_rotation(position, (210, 330)));
    }

    let engine: Engine = Engine { node_positions, node_rotations };

    Python::with_gil(|py| {
        py.allow_threads(|| {
            run_bevy_app(engine);
        });
    });

    Ok(())
}

#[pymodule]
fn reply_drone(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start, m)?)?;
    Ok(())
}
