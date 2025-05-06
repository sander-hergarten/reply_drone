mod depthmap;
mod input;
mod backend;
mod reader;
mod helper;
mod commands;
mod constants;
mod screenshot;
mod components;

mod types;
use std::sync::{Mutex, OnceLock};

use types::*;

use commands::{EngineCommands, EngineResponses};
use crossbeam_channel::Sender;

use bevy::prelude::*;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use smooth_bevy_cameras::{
    controllers::unreal::UnrealCameraPlugin,
    LookTransformPlugin,
};

static COMMAND_SENDER: OnceLock<Sender<EngineCommands>> = OnceLock::new();

fn run_bevy_app() {
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
        // .add_systems(Update, (input::toggle_prepass_view, input::take_depth_screenshot_on_tab))
        .add_systems(Update, backend::command_processing_system)
        .run();
    println!("[Rust] Bevy app finished.");
}

#[pyfunction]
fn start() -> PyResult<()> {
    let (cmd_tx, _) = crossbeam_channel::unbounded();
    COMMAND_SENDER.set(cmd_tx)
            .map_err(|_| PyRuntimeError::new_err("x is negative"))?;

    Python::with_gil(|py| {
        py.allow_threads(|| {
            run_bevy_app();
        });
    });

    Ok(())
}

#[pyfunction]
fn add_node(position: Position, rotation: Rotation) -> PyResult<u32> {
    let (response_tx, response_rx) = crossbeam_channel::bounded(1);

    let sender = COMMAND_SENDER.get()
            .ok_or_else(|| PyRuntimeError::new_err("Engine not initialized"))?;

    sender.send(EngineCommands::AddNode { 
        position, rotation, response_tx: Some(response_tx)
    }).map_err(|e| PyRuntimeError::new_err(format!("Failed to send command: {}", e)))?;

    let result = response_rx.recv()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to send command: {}", e)))?;

    match result {
        EngineResponses::AddNodeResponse { id } => Ok(id),
        _ => Err(PyRuntimeError::new_err("Unexpected response type"))
    }
}

#[pyfunction]
fn add_random_node() -> PyResult<u32> {
    let mesh_bounds: [(i32, i32); 3] = reader::read_glb_bounds("centered.glb");
    let position: Position = helper::random_position(mesh_bounds);
    add_node(position, helper::generate_camera_rotation(position, (210, 330)))
}

#[pyfunction]
fn get_engine() -> PyResult<()> {

    Ok(())
}

#[pyfunction]
fn get_full_graph_state() -> PyResult<()> {
    let (response_tx, response_rx) = crossbeam_channel::bounded(1);

    let sender = COMMAND_SENDER.get()
            .ok_or_else(|| PyRuntimeError::new_err("Engine not initialized"))?;

    sender.send(EngineCommands::GetFullGraphState { 
        response_tx: Some(response_tx)
    }).map_err(|e| PyRuntimeError::new_err(format!("Failed to send command: {}", e)))?;

    let result = response_rx.recv()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to send command: {}", e)))?;

    match result {
        EngineResponses::FullGraphStateResponse { full_graph_state } => {
            println!("Got full_graph_state {}", full_graph_state.len());
        },
        _ => {
            return Err(PyRuntimeError::new_err("Unexpected response type"));
        }
    }
     
    Ok(())
}

#[pymodule]
fn reply_drone(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start, m)?)?;
    m.add_function(wrap_pyfunction!(add_node, m)?)?;
    m.add_function(wrap_pyfunction!(add_random_node, m)?)?;
    m.add_function(wrap_pyfunction!(get_engine, m)?)?;
    m.add_function(wrap_pyfunction!(get_full_graph_state, m)?)?;
    Ok(())
}
