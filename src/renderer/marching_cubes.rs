use super::types::Model;
use crate::las_converter::heightmap::Heightmap;
use lin_alg::f32::Vec3;
use log::info;
use mcubes::{MarchingCubes, MeshSide};

pub fn of_heightmap(heightmap: &Heightmap<f32>, z_steps: usize) -> Model {
    info!(
        "Meshifying heightmap of size {} {}",
        heightmap.width, heightmap.height
    );

    let depth = z_steps;

    let mut values = Vec::new();

    let insert_empty_height = |values: &mut Vec<_>| {
        for _y in 0..(heightmap.height + 2) {
            for _x in 0..(heightmap.width + 2) {
                values.push(0.);
            }
        }
    };

    let insert_empty_row = |values: &mut Vec<_>| {
        for _x in 0..(heightmap.width + 2) {
            values.push(0.);
        }
    };

    let insert_empty_cell = |values: &mut Vec<_>| {
        values.push(0.);
    };

    // We pad our data with empty cells so that marching cubes produces a bounding volume rather
    // than just the skin of our terrain.

    insert_empty_height(&mut values);

    for z in 0..depth {
        insert_empty_row(&mut values);

        for y in 0..heightmap.height {
            insert_empty_cell(&mut values);
            for x in 0..heightmap.width {
                let sz = (z as f32) / (depth as f32);
                let test = heightmap[(x, y)] >= sz;
                values.push(if test { 1. } else { 0. });
            }
            insert_empty_cell(&mut values);
        }

        insert_empty_row(&mut values);
    }

    insert_empty_height(&mut values);

    info!("Preparing marching cubes");

    let cell_depth = 1. / (depth as f32);

    let extent = (
        heightmap.width + 2 as usize,
        heightmap.height + 2 as usize,
        depth + 2 as usize,
    );

    info!("{:?}", extent);
    info!("{}", cell_depth);

    let mc = MarchingCubes::new(
        extent,
        (1., 1., 1.),
        (extent.0 as f32, extent.1 as f32, extent.2 as f32),
        Vec3::new(0., 0., 0.),
        values,
        1.0,
    )
    .unwrap();

    info!("Running marching cubes");

    let mesh = mc.generate(MeshSide::OutsideOnly);

    info!("Triangulating mesh");

    let vertices: Vec<[f32; 3]> = mesh
        .vertices
        .into_iter()
        .map(|v| v.posit.to_arr())
        .collect();

    let indices = mesh.indices;

    //let mut triangles: Vec<Triangle> = Vec::new();
    //for triangle in indices.chunks(3) {
    //    triangles.push([
    //        vertices[triangle[1]],
    //        vertices[triangle[0]],
    //        vertices[triangle[2]],
    //    ]);
    //}

    let mut result = Model {
        triangles: indices.chunks(3).map(|x| [x[1], x[0], x[2]]).collect(),
        vertices,
    };

    result.scale((
        heightmap.width as f32,
        heightmap.height as f32,
        heightmap.scale_z,
    ));

    result
}
