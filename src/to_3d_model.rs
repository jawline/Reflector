use crate::heightmap::Heightmap;
use bevy::math::Vec3;
use log::info;

/// Add a border along the y axis at a fixed x (x should either be 0 or heightmap.height - 1)
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

/// Add a border along the x axis at a fixed y (y should either be 0 or heightmap.width - 1)
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

/// Add four vertices for a base to the model.
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

/// Compute the normal as the cross product of (v1 - v2) nd (v1- v3). Depending on the direction
/// the normal might need to be negated.
fn compute_normal(
    (x1, y1): (usize, usize),
    (x2, y2): (usize, usize),
    (x3, y3): (usize, usize),
    vertices: &[[f32; 3]],
    heightmap: &Heightmap<f64>,
) -> Vec3 {
    let off1 = heightmap.offset((x1, y1));
    let off2 = heightmap.offset((x2, y2));
    let off3 = heightmap.offset((x3, y3));

    let va: Vec3 = vertices[off1].into();
    let vb: Vec3 = vertices[off2].into();
    let vc: Vec3 = vertices[off3].into();
    (va - vc).cross(va - vb)
}

pub struct Model {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

impl Model {
    pub fn scale(&mut self, (scale_x, scale_y, scale_z): (f32, f32, f32)) {
        for vertex in &mut self.vertices {
            vertex[0] = vertex[0] * scale_x;
            vertex[1] = vertex[1] * scale_y;
            vertex[2] = vertex[2] * scale_z;
        }
    }

    pub fn of_heightmap(heightmap: &Heightmap<f64>) -> Self {
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
                    sum += -compute_normal((x, y), (x, y - 1), (x - 1, y), &vertices, &heightmap);
                    total += 1;
                }

                if x < heightmap.width - 1 && y < heightmap.height - 1 {
                    sum += -compute_normal((x, y), (x, y + 1), (x + 1, y), &vertices, &heightmap);
                    total += 1;
                }

                if x > 0 && y < heightmap.height - 1 {
                    sum += compute_normal((x, y), (x, y + 1), (x - 1, y), &vertices, &heightmap);
                    total += 1;
                }

                if x < heightmap.width - 1 && y > 0 {
                    sum += -compute_normal((x, y), (x + 1, y), (x, y - 1), &vertices, &heightmap);
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

        // Compute indices
        let mut indices: Vec<u32> = Vec::new();

        for y in 0..(heightmap.height - 1) {
            for x in 0..(heightmap.width - 1) {
                let xoff = heightmap.offset((x, y)) as u32;
                let next_y_xoff = heightmap.offset((x, y + 1)) as u32;
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
            let y3 = heightmap.offset((x, y));
            let y4 = heightmap.offset((x, y + 1));

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
            let y3 = heightmap.offset((x, y));
            let y4 = heightmap.offset((x, y + 1));

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
            let x3 = heightmap.offset((x, y));
            let x4 = heightmap.offset((x + 1, y));

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
            let x3 = heightmap.offset((x, y));
            let x4 = heightmap.offset((x + 1, y));

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

        let mut result = Model {
            vertices,
            normals,
            uvs,
            indices,
        };
        let model_scale_factor = (1. / heightmap.pixels_per_distance_unit) as f32;
        result.scale((model_scale_factor, 1., model_scale_factor));
        result
    }
}
