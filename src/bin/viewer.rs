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
use rust_las_printer::heightmap::Heightmap;
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

fn add_y_border(
    x: usize,
    norm: [f32; 3],
    vertices: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    heightmap: &Heightmap<f64>,
) -> usize {
    let offset = vertices.len();

    for y in 0..heightmap.height {
        vertices.push([x as f32, 0., y as f32]);
        normals.push(norm.clone());
        uvs.push([
            x as f32 / heightmap.width as f32,
            y as f32 / heightmap.height as f32,
        ]);
    }

    offset
}

fn add_x_border(
    y: usize,
    norm: [f32; 3],
    vertices: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    heightmap: &Heightmap<f64>,
) -> usize {
    let offset = vertices.len();

    for x in 0..heightmap.width {
        vertices.push([x as f32, 0., y as f32]);
        normals.push(norm.clone());
        uvs.push([
            x as f32 / heightmap.width as f32,
            y as f32 / heightmap.height as f32,
        ]);
    }

    offset
}

fn add_base(
    vertices: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    heightmap: &Heightmap<f64>,
) -> usize {
    let offset = vertices.len();

    vertices.push([0., 0., 0.]);
    normals.push([0., -1., 0.]);
    uvs.push([0., 0.]);

    vertices.push([(heightmap.width - 1) as f32, 0., 0.]);
    normals.push([0., -1., 0.]);
    uvs.push([0., 0.]);

    vertices.push([0., 0., (heightmap.height - 1) as f32]);
    normals.push([0., -1., 0.]);
    uvs.push([0., 0.]);

    vertices.push([
        (heightmap.height - 1) as f32,
        0.,
        (heightmap.height - 1) as f32,
    ]);
    normals.push([0., -1., 0.]);
    uvs.push([0., 0.]);

    offset
}

fn offset((x, y): (usize, usize), (width, _height): (usize, usize)) -> usize {
    (y * width) + x
}

fn compute_normal(
    (x1, y1): (usize, usize),
    (x2, y2): (usize, usize),
    (x3, y3): (usize, usize),
    vertices: &[[f32; 3]],
    (width, height): (usize, usize),
) -> Vec3 {
    let off1 = offset((x1, y1), (width, height));
    let off2 = offset((x2, y2), (width, height));
    let off3 = offset((x3, y3), (width, height));

    let va: Vec3 = vertices[off1].into();
    let vb: Vec3 = vertices[off2].into();
    let vc: Vec3 = vertices[off3].into();
    (va - vc).cross(va - vb)
}

fn heightmap_to_mesh_and_image(heightmap: &Heightmap<f64>) -> (Mesh, Image) {
    let mut vertices = Vec::new();
    let mut uvs = Vec::new();

    info!(
        "Meshifying heightmap of size {} {}",
        heightmap.width, heightmap.height
    );

    // The body of the mesh
    for y in 0..heightmap.height {
        for x in 0..heightmap.width {
            vertices.push([x as f32, heightmap[(x, y)] as f32, y as f32]);
            uvs.push([
                x as f32 / heightmap.width as f32,
                y as f32 / heightmap.height as f32,
            ]);
        }
    }

    let mut normals: Vec<[f32; 3]> = Vec::new();

    for y in 0..heightmap.height {
        for x in 0..heightmap.width {
            let mut sum = Vec3::new(0., 0., 0.);
            let mut total = 0;

            // TODO: Area weight the normal average by multiplying each sum addition by the area of
            // it and making total the total area of all normals rather than a count.

            if x > 0 && y > 0 {
                sum += -compute_normal(
                    (x, y),
                    (x, y - 1),
                    (x - 1, y),
                    &vertices,
                    (heightmap.width, heightmap.height),
                );
                total += 1;
            }

            if x < heightmap.width - 1 && y < heightmap.height - 1 {
                sum += -compute_normal(
                    (x, y),
                    (x, y + 1),
                    (x + 1, y),
                    &vertices,
                    (heightmap.width, heightmap.height),
                );
                total += 1;
            }

            if x > 0 && y < heightmap.height - 1 {
                sum += compute_normal(
                    (x, y),
                    (x, y + 1),
                    (x - 1, y),
                    &vertices,
                    (heightmap.width, heightmap.height),
                );
                total += 1;
            }

            if x < heightmap.width - 1 && y > 0 {
                sum += -compute_normal(
                    (x, y),
                    (x + 1, y),
                    (x, y - 1),
                    &vertices,
                    (heightmap.width, heightmap.height),
                );
                total += 1;
            }

            normals.push((sum / total as f32).into());
        }
    }

    // Add some border and a base
    let left_row_offset = add_y_border(
        0,
        [1., 0., 0.],
        &mut vertices,
        &mut normals,
        &mut uvs,
        heightmap,
    );
    let right_row_offset = add_y_border(
        heightmap.width - 1,
        [-1., 0., 0.],
        &mut vertices,
        &mut normals,
        &mut uvs,
        heightmap,
    );
    let bottom_offset = add_x_border(
        0,
        [0., 1., 0.],
        &mut vertices,
        &mut normals,
        &mut uvs,
        heightmap,
    );
    let top_offset = add_x_border(
        heightmap.height - 1,
        [0., -1., 0.],
        &mut vertices,
        &mut normals,
        &mut uvs,
        heightmap,
    );
    let base_offset = add_base(&mut vertices, &mut normals, &mut uvs, heightmap);

    // TODO base

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);

    let mut indices: Vec<u32> = Vec::new();

    for y in 0..(heightmap.height - 1) {
        for x in 0..(heightmap.width - 1) {
            let xoff = ((y * heightmap.width) + x) as u32;
            let next_y_xoff = (((y + 1) * heightmap.width) + x) as u32;
            indices.push(xoff);
            indices.push(next_y_xoff);
            indices.push(xoff + 1);
            indices.push(next_y_xoff);
            indices.push(next_y_xoff + 1);
            indices.push(xoff + 1);
        }
    }

    // Add Y borders
    // This is tricky to generalize because the order of the indices effects the direction the
    // vertices will appear from
    for y in 0..(heightmap.height - 1) {
        let x = 0;
        let y1 = left_row_offset + y;
        let y2 = left_row_offset + y + 1;
        let y3 = ((y * heightmap.width) + x) as u32;
        let y4 = (((y + 1) * heightmap.width) + x) as u32;

        indices.push(y1 as u32);
        indices.push(y2 as u32);
        indices.push(y4 as u32);

        indices.push(y4 as u32);
        indices.push(y3 as u32);
        indices.push(y1 as u32);
    }

    for y in 0..(heightmap.height - 1) {
        let x = heightmap.width - 1;
        let y1 = right_row_offset + y;
        let y2 = right_row_offset + y + 1;
        let y3 = ((y * heightmap.width) + x) as u32;
        let y4 = (((y + 1) * heightmap.width) + x) as u32;

        indices.push(y4 as u32);
        indices.push(y2 as u32);
        indices.push(y1 as u32);

        indices.push(y1 as u32);
        indices.push(y3 as u32);
        indices.push(y4 as u32);
    }

    // Add X borders
    for x in 0..(heightmap.width - 1) {
        let y = 0;
        let x1 = bottom_offset + x;
        let x2 = bottom_offset + x + 1;
        let x3 = ((y * heightmap.width) + x) as u32;
        let x4 = ((y * heightmap.width) + x + 1) as u32;

        indices.push(x4 as u32);
        indices.push(x2 as u32);
        indices.push(x1 as u32);

        indices.push(x1 as u32);
        indices.push(x3 as u32);
        indices.push(x4 as u32);
    }

    for x in 0..(heightmap.width - 1) {
        let y = heightmap.height - 1;
        let x1 = top_offset + x;
        let x2 = top_offset + x + 1;
        let x3 = ((y * heightmap.width) + x) as u32;
        let x4 = ((y * heightmap.width) + x + 1) as u32;

        indices.push(x1 as u32);
        indices.push(x2 as u32);
        indices.push(x4 as u32);

        indices.push(x4 as u32);
        indices.push(x3 as u32);
        indices.push(x1 as u32);
    }

    // Add base
    indices.push(base_offset as u32);
    indices.push(base_offset as u32 + 1);
    indices.push(base_offset as u32 + 2);

    indices.push(base_offset as u32 + 3);
    indices.push(base_offset as u32 + 1);
    indices.push(base_offset as u32 + 2);

    mesh.set_indices(Some(Indices::U32(indices)));

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
        // The default cascade config is designed to handle large scenes.
        // As this example has a much smaller world, we can tighten the shadow
        // bounds for better visual quality.
        cascade_shadow_config: CascadeShadowConfigBuilder {
            minimum_distance: 1.,
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
