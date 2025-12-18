use bevy::{
    asset::{Asset, Handle},
    color::LinearRgba,
    image::Image,
    pbr::Material,
    reflect::TypePath,
    render::{
        alpha::AlphaMode,
        render_resource::{AsBindGroup, ShaderType},
    },
    shader::ShaderRef,
};

use crate::constants::{MATERIAL_SHADER_ASSET_PATH, PREPASS_SHADER_ASSET_PATH};

#[derive(Debug, Clone, Default, ShaderType, Copy)]
pub struct ShowPrepassSettings {
    pub show_depth: u32,
    pub show_normals: u32,
    pub show_motion_vectors: u32,
    pub padding_1: u32,
    pub padding_2: u32,
}

// This shader simply loads the prepass texture and outputs it directly
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone, Copy)]
pub struct PrepassOutputMaterial {
    #[uniform(0)]
    pub settings: ShowPrepassSettings,
}

impl Material for PrepassOutputMaterial {
    fn fragment_shader() -> ShaderRef {
        PREPASS_SHADER_ASSET_PATH.into()
    }

    // This needs to be transparent in order to show the scene behind the mesh
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

// will be passed to your shader
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CustomMaterial {
    #[uniform(0)]
    pub color: LinearRgba,
    #[texture(1)]
    #[sampler(2)]
    pub color_texture: Option<Handle<Image>>,
    pub alpha_mode: AlphaMode,
}

/// function will also be used by the prepass
impl Material for CustomMaterial {
    fn fragment_shader() -> ShaderRef {
        MATERIAL_SHADER_ASSET_PATH.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }
}
