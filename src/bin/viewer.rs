use bevy::input::keyboard::KeyCode;
use bevy::input::keyboard::KeyboardInput;
use bevy::input::ButtonState;
use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::pbr::CascadeShadowConfigBuilder;
use bevy::prelude::*;
use bevy::render::mesh::Indices;
use bevy::render::render_resource::Extent3d;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::render::render_resource::TextureDimension;
use bevy::render::render_resource::TextureFormat;
use bevy_panorbit_camera::PanOrbitCamera;
use bevy_panorbit_camera::PanOrbitCameraPlugin;
use clap::Parser;
use env_logger;
use log::info;
use rust_las_printer::{heightmap::Heightmap, to_3d_model::Model};
use std::f32::consts::PI;
use std::{
    fs::File,
    io::{BufReader, Read},
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    input_path: String,
}

fn main() {
    env_logger::init();
    App::new()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup)
        .add_system(print_keyboard_event_system)
        .add_plugin(PanOrbitCameraPlugin)
        .add_plugin(WireframePlugin)
        .run();
}

fn bytes(path: &str) -> Vec<u8> {
    let f = File::open(path).unwrap();
    let mut reader = BufReader::new(f);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).unwrap();
    buffer
}

fn heightmap_to_mesh_and_image(heightmap: &Heightmap<f64>) -> (Mesh, Image) {
    let model = Model::of_heightmap(&heightmap);

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, model.vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, model.normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, model.uvs);
    mesh.set_indices(Some(Indices::U32(model.indices)));

    let size = Extent3d {
        width: heightmap.width as u32,
        height: heightmap.height as u32,
        ..default()
    };

    let image_data: Vec<u8> = heightmap.data.iter().map(|_| 255).collect();

    let image = Image::new_fill(
        size,
        TextureDimension::D2,
        &image_data,
        TextureFormat::R8Unorm,
    );

    (mesh, image)
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {

    let args = Args::parse();
    let bytes = bytes(&args.input_path);
    let heightmap: Heightmap<f64> = postcard::from_bytes(&bytes).unwrap();
    let (mesh, heightmap_texture) = heightmap_to_mesh_and_image(&heightmap);
    let (start_x, start_y) = (heightmap.width as f32 / 2., heightmap.height as f32 / 2.);

    commands.spawn(PbrBundle {
        mesh: meshes.add(mesh),
        material: materials.add(StandardMaterial {
            base_color_texture: Some(images.add(heightmap_texture)),
            ..default()
        }),
        transform: Transform::from_xyz(0., 0., 0.),
        ..default()
    });

    commands.insert_resource(AmbientLight {
        color: Color::ORANGE_RED,
        brightness: 0.3,
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        transform: Transform {
            translation: Vec3::new(0.0, 200., heightmap.height as f32),
            rotation: (Quat::from_rotation_z(-PI / 1.) + Quat::from_rotation_x(-PI / 1.5)),
            ..default()
        },
        cascade_shadow_config: CascadeShadowConfigBuilder {
            minimum_distance: 0.01,
            maximum_distance: 10000.,
            ..default()
        }
        .into(),
        ..default()
    });

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(start_x, 500., start_x)
                .looking_at((start_x, 0., start_y).into(), Vec3::Y),
            ..default()
        },
        PanOrbitCamera::default(),
    ));
}

fn print_keyboard_event_system(
    mut keyboard_input_events: EventReader<KeyboardInput>,
    mut wireframe_config: ResMut<WireframeConfig>,
) {
    for event in keyboard_input_events.iter() {
        info!("{:?}", event);
        match event {
            KeyboardInput {
                scan_code: 17,
                key_code: Some(KeyCode::W),
                state: ButtonState::Pressed,
            } => wireframe_config.global = !wireframe_config.global,
            _ => {}
        }
    }
}
