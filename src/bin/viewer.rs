use bevy::input::keyboard::KeyboardInput;
use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::{
        fxaa::Fxaa,
        prepass::{DeferredPrepass, DepthPrepass},
    },
    input::ButtonState,
    pbr::CascadeShadowConfigBuilder,
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use clap::Parser;
use env_logger;
use log::info;
use rust_las_printer::{heightmap::Heightmap, to_3d_model::Model};
use std::fs::read;

use serde_pickle::DeOptions;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    input_path: String,
}

fn main() {
    env_logger::init();
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, print_keyboard_event_system)
        .add_plugins((PanOrbitCameraPlugin, WireframePlugin { ..default() }))
        .run();
}

fn heightmap_to_mesh_and_image(heightmap: &Heightmap<f32>) -> Mesh {
    let model = Model::of_heightmap(&heightmap);

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );

    let vertices_in_order: Vec<[f32; 3]> = model.triangles.iter().flat_map(|x| x.clone()).collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices_in_order);
    mesh.compute_flat_normals();

    mesh
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut _images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let args = Args::parse();

    let heightmap = read(&args.input_path).unwrap();
    let heightmap: Heightmap<Option<f32>> =
        serde_pickle::from_slice(&heightmap, DeOptions::new()).unwrap();
    let heightmap = heightmap.fill_none_with_zero_and_add_base(0.0, 0.0);
    let mesh = heightmap_to_mesh_and_image(&heightmap);
    let (start_x, start_y) = (heightmap.width as f32 / 2., heightmap.height as f32 / 2.);

    let base_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1., 0., 0.0),
        perceptual_roughness: 0.01,
        metallic: 0.0,
        ..default()
    });

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(base_material),
        Transform::from_xyz(0., 0., 0.),
    ));

    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 1200.,
        affects_lightmapped_meshes: true,
    });

    commands.spawn((
        DirectionalLight {
            illuminance: 350.,
            shadows_enabled: true,
            ..default()
        },
        CascadeShadowConfigBuilder {
            num_cascades: 3,
            maximum_distance: 10.0,
            ..default()
        }
        .build(),
        Transform::from_xyz(start_x, 500., start_y)
            .looking_at((start_x, 0., start_y).into(), Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(start_x, 150., start_y)
            .looking_at((start_x, 0., start_y).into(), Vec3::Y),
        Camera {
            hdr: false,
            ..default()
        },
        Msaa::Off,
        DepthPrepass,
        DeferredPrepass,
        Fxaa::default(),
        PanOrbitCamera::default(),
    ));
}
fn print_keyboard_event_system(
    mut keyboard_input_events: EventReader<KeyboardInput>,
    mut wireframe_config: ResMut<WireframeConfig>,
) {
    for event in keyboard_input_events.read() {
        info!("{:?}", event);
        match event.state {
            ButtonState::Pressed => {
                wireframe_config.global = !wireframe_config.global;
            }
            ButtonState::Released => (),
        };
    }
}
