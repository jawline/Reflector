use bevy::input::keyboard::KeyboardInput;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use itertools::iproduct;
use rust_las_printer::heightmap::Heightmap;
use std::{
    fs::File,
    io::{BufReader, Read},
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    input_path: String,
}

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

fn heightmap_to_mesh(heightmap: &Heightmap<f64>) -> Mesh {
    let mut vertices = Vec::new();

    for (x, y) in iproduct!(0..heightmap.width, 0..heightmap.height) {
        vertices.push([x as f32, y as f32, heightmap[(x, y)] as f32]);
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);

    mesh
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {

    let args = Args::parse();
    let bytes = bytes(&args.input_path);
    let heightmap: Heightmap<f64> = postcard::from_bytes(&bytes).unwrap();

    let mesh = heightmap_to_mesh(&heightmap);
    commands.spawn(PbrBundle {
        mesh: meshes.add(mesh),
        material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
        transform: Transform::from_xyz(0., 0., 0.),
        ..default()
    });

    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

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
