use gltf::Gltf;
use std::fs;
use std::path::Path;

/// Returns the axis-aligned bounding box (AABB) of a GLB file.
/// If there is any error, it prints an error message and returns [(0,0), (0,0), (0,0)].
pub fn read_glb_bounds<P: AsRef<Path>>(path: P) -> [(i32, i32); 3] {
    let data = match fs::read(&path) {
        Ok(data) => data,
        Err(_) => {
            println!("File not found: {:?}", path.as_ref());
            return [(0, 0), (0, 0), (0, 0)];
        }
    };

    let glb = match Gltf::from_slice(&data) {
        Ok(glb) => glb,
        Err(_) => {
            println!("Failed to parse GLB file: {:?}", path.as_ref());
            return [(0, 0), (0, 0), (0, 0)];
        }
    };

    let blob = match glb.blob.as_ref() {
        Some(blob) => blob,
        None => {
            println!("Missing binary blob in GLB file.");
            return [(0, 0), (0, 0), (0, 0)];
        }
    };

    let mut min = [i32::MAX, i32::MAX, i32::MAX];
    let mut max = [i32::MIN, i32::MIN, i32::MIN];

    for mesh in glb.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|_| Some(&blob[..])); // <- just call it directly

            if let Some(positions) = reader.read_positions() {
                for pos in positions {
                    for i in 0..3 {
                        let value = pos[i];
                        let value = value.round() as i32;
                        if value < min[i] {
                            min[i] = value;
                        }
                        if value > max[i] {
                            max[i] = value;
                        }
                    }
                }
            }
        }
    }

    [
        (min[0], max[0]),
        (min[1], max[1]),
        (min[2], max[2]),
    ]
}
