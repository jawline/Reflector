use bevy::input::keyboard::KeyboardInput;
use bevy::prelude::*;
use itertools::iproduct;
use rust_las_printer::heightmap::Heightmap;
use std::{
    fs::File,
    io::{BufReader, Read},
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup)
        .add_system(print_keyboard_event_system)
        .run();
}

fn bytes(path: &str) -> Vec<u8> {
    let f = File::open(path).unwrap();
    let mut reader = BufReader::new(f);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).unwrap();
    buffer
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let bytes = bytes("/home/blake/Downloads/test.bin");
    let heightmap: Heightmap<f64> = postcard::from_bytes(&bytes).unwrap();

    for (x, y) in iproduct!(0..heightmap.width, 0..heightmap.height) {
        commands.spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
            transform: Transform::from_xyz(x as f32, y as f32, 0.),
            ..default()
        });
    }

    // light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });
    // camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
}

fn print_keyboard_event_system(mut keyboard_input_events: EventReader<KeyboardInput>) {
    for event in keyboard_input_events.iter() {
        info!("{:?}", event);
    }
}
