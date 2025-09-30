use super::types::{Model, Point, Triangle};
use crate::las_converter::heightmap::Heightmap;
use log::info;

/// Add a border along the y axis at a fixed x (x should either be 0 or heightmap.height - 1)
fn add_y_border(x: usize, vertices: &mut Vec<Point>, heightmap: &Heightmap<f32>) -> usize {
    let offset = vertices.len();

    for y in 0..heightmap.height {
        vertices.push([-(x as f32), y as f32, 0.]);
    }

    offset
}

/// Add a border along the x axis at a fixed y (y should either be 0 or heightmap.width - 1)
fn add_x_border(y: usize, vertices: &mut Vec<Point>, heightmap: &Heightmap<f32>) -> usize {
    let offset = vertices.len();

    for x in 0..heightmap.width {
        vertices.push([-(x as f32), y as f32, 0.]);
    }

    offset
}

/// Add four vertices for a base to the model.
fn add_base(vertices: &mut Vec<Point>, heightmap: &Heightmap<f32>) -> usize {
    let offset = vertices.len();

    vertices.push([0., 0., 0.]);

    vertices.push([-((heightmap.width - 1) as f32), 0., 0.]);

    vertices.push([0., (heightmap.height - 1) as f32, 0.]);

    vertices.push([
        -((heightmap.height - 1) as f32),
        (heightmap.height - 1) as f32,
        0.,
    ]);

    offset
}

pub fn of_heightmap(heightmap: &Heightmap<f32>) -> Model {
    let mut vertices = Vec::new();

    info!(
        "Meshifying heightmap of size {} {}",
        heightmap.width, heightmap.height
    );

    // The body of the mesh
    for y in 0..heightmap.height {
        for x in 0..heightmap.width {
            vertices.push([-(x as f32), y as f32, heightmap[(x, y)] as f32]);
        }
    }

    // Add some border and a base
    let left_row_offset = add_y_border(0, &mut vertices, heightmap);
    let right_row_offset = add_y_border(heightmap.width - 1, &mut vertices, heightmap);
    let bottom_offset = add_x_border(0, &mut vertices, heightmap);
    let top_offset = add_x_border(heightmap.height - 1, &mut vertices, heightmap);
    let base_offset = add_base(&mut vertices, heightmap);

    // Compute indices
    let mut triangles: Vec<Triangle> = Vec::new();

    let mut add_tri = |x, y, z| {
        triangles.push([x, y, z]);
    };

    for y in 0..(heightmap.height - 1) {
        for x in 0..(heightmap.width - 1) {
            let xoff = heightmap.offset((x, y)) as usize;
            let next_y_xoff = heightmap.offset((x, y + 1)) as usize;
            add_tri(xoff, next_y_xoff, xoff + 1);
            add_tri(next_y_xoff, next_y_xoff + 1, xoff + 1);
        }
    }

    // Add Y borders
    // This is tricky to generalize because the order of the indices effects the direction the
    // vertices will appear from
    for y in 0..(heightmap.height - 1) {
        let x = 0;
        let y1 = left_row_offset + y;
        let y2 = left_row_offset + y + 1;
        let y3 = heightmap.offset((x, y));
        let y4 = heightmap.offset((x, y + 1));

        add_tri(y1 as usize, y2 as usize, y4 as usize);
        add_tri(y4 as usize, y3 as usize, y1 as usize);
    }

    for y in 0..(heightmap.height - 1) {
        let x = heightmap.width - 1;
        let y1 = right_row_offset + y;
        let y2 = right_row_offset + y + 1;
        let y3 = heightmap.offset((x, y));
        let y4 = heightmap.offset((x, y + 1));

        add_tri(y4 as usize, y2 as usize, y1 as usize);
        add_tri(y1 as usize, y3 as usize, y4 as usize);
    }

    // Add X borders
    for x in 0..(heightmap.width - 1) {
        let y = 0;
        let x1 = bottom_offset + x;
        let x2 = bottom_offset + x + 1;
        let x3 = heightmap.offset((x, y));
        let x4 = heightmap.offset((x + 1, y));

        add_tri(x4 as usize, x2 as usize, x1 as usize);
        add_tri(x1 as usize, x3 as usize, x4 as usize);
    }

    for x in 0..(heightmap.width - 1) {
        let y = heightmap.height - 1;
        let x1 = top_offset + x;
        let x2 = top_offset + x + 1;
        let x3 = heightmap.offset((x, y));
        let x4 = heightmap.offset((x + 1, y));

        add_tri(x1 as usize, x2 as usize, x4 as usize);
        add_tri(x4 as usize, x3 as usize, x1 as usize);
    }

    // Add base
    add_tri(
        base_offset as usize,
        base_offset as usize + 1,
        base_offset as usize + 2,
    );
    add_tri(
        base_offset as usize + 3,
        base_offset as usize + 1,
        base_offset as usize + 2,
    );

    let mut result = Model {
        vertices,
        triangles,
    };

    result.scale((1., 1., heightmap.scale_z));
    result
}
