use bevy::input::keyboard::KeyboardInput;
use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::prelude::*;
use bevy::render::{mesh::Indices, render_resource::PrimitiveTopology};
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
use serde_pickle::DeOptions;
use std::f32::consts::PI;
use std::fs::read;

use rust_las_printer::{las_converter::heightmap::Heightmap, renderer::to_3d_model};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    input_path: String,

    #[arg(short, long)]
    mode: to_3d_model::Mode,
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

fn heightmap_to_mesh_and_image(heightmap: &Heightmap<f32>, mode: &to_3d_model::Mode) -> Mesh {
    let model = to_3d_model::of_heightmap(&heightmap, mode);

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );

    let vertices: Vec<[f32; 3]> = model.vertices.into_iter().collect();
    let indices: Vec<u32> = model
        .triangles
        .into_iter()
        .flat_map(|x| x)
        .map(|x| x as u32)
        .collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_smooth_normals();

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
    let heightmap = heightmap.fill_none_with_zero_and_add_base(0.0, 0.1); // TODO: Renormalize
    let mesh = heightmap_to_mesh_and_image(&heightmap, &args.mode);
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
        brightness: 2000.,
        affects_lightmapped_meshes: true,
    });

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::OVERCAST_DAY,
            shadows_enabled: true,
            ..default()
        },
        Transform {
            translation: Vec3::new(start_x, 100., start_y),
            rotation: Quat::from_rotation_x(-PI / 4.),
            ..default()
        },
        // The default cascade config is designed to handle large scenes.
        // As this example has a much smaller world, we can tighten the shadow
        // bounds for better visual quality.
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 4.0,
            maximum_distance: 100000.0,
            ..default()
        }
        .build(),
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
