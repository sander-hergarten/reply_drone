use rand::Rng;

pub type Position = [i32; 3];
pub type Rotation = [i32; 3];

/// Generates a random position with bounds.
pub fn random_position(
    bounds: [(i32, i32); 3]
) -> Position {
    let mut rng = rand::rng();
    [
        rng.random_range(bounds[0].0..=bounds[0].1),
        rng.random_range(bounds[1].0..=bounds[1].1),
        rng.random_range(bounds[2].0..=bounds[2].1),
    ]
}

/// Generates a random camera rotation for looking downward towards the center.
pub fn generate_camera_rotation(
    position: Position,
    pitch_bounds: (i32, i32), // e.g., (210, 330)
) -> Rotation {
    let mut rng = rand::rng();

    // Random pitch within bounds (downward angles)
    let pitch = rng.random_range(pitch_bounds.0..=pitch_bounds.1);

    // Calculate yaw to face the origin (0, 0, 0)
    let x = position[0] as f32;
    let z = position[2] as f32;

    let yaw_rad = z.atan2(x); // z first, x second
    let yaw_deg = yaw_rad.to_degrees();
    let yaw = ((yaw_deg + 360.0) % 360.0).round() as i32;

    // Roll is always 0
    let roll = 0;

    [pitch, yaw, roll]
}
