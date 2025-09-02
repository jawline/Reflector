use crate::heightmap::Heightmap;
use bevy::math::Vec3;
use log::info;

/// Compute the normal as the cross product of (v1 - v2) nd (v1- v3). Depending on the direction
/// the normal might need to be negated.
fn compute_normal(
    (x1, y1): (usize, usize),
    (x2, y2): (usize, usize),
    (x3, y3): (usize, usize),
    vertices: &[[f32; 3]],
    heightmap: &Heightmap<f32>,
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

const VERTEX_STRIDE: usize = 8;

impl Model {
    pub fn scale(&mut self, (scale_x, scale_y, scale_z): (f32, f32, f32)) {
        for vertex in &mut self.vertices {
            vertex[0] = vertex[0] * scale_x;
            vertex[1] = vertex[1] * scale_y;
            vertex[2] = vertex[2] * scale_z;
        }
    }

    pub fn of_heightmap(heightmap: &Heightmap<f32>) -> Self {
        let mut vertices = Vec::new();
        let mut uvs = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();

        info!(
            "Meshifying heightmap of size {} {}",
            heightmap.width, heightmap.height
        );

        // The body of the mesh
        for y in 0..heightmap.height {
            for x in 0..heightmap.width {
                vertices.push([-(x as f32), y as f32, heightmap[(x, y)] as f32]);
                vertices.push([-((x + 1) as f32), y as f32, heightmap[(x, y)] as f32]);
                vertices.push([-(x as f32), (y + 1) as f32, heightmap[(x, y)] as f32]);
                vertices.push([-((x + 1) as f32), (y + 1) as f32, heightmap[(x, y)] as f32]);

                vertices.push([-(x as f32), y as f32, 0.]);
                vertices.push([-((x + 1) as f32), y as f32, 0.]);
                vertices.push([-(x as f32), (y + 1) as f32, 0.]);
                vertices.push([-((x + 1) as f32), (y + 1) as f32, 0.]);

                for i in 0..8 {
                    uvs.push([
                        x as f32 / heightmap.width as f32,
                        y as f32 / heightmap.height as f32,
                    ]);
                }

                normals.push([1., -1., 1.]);
                normals.push([-1., -1., 1.]);
                normals.push([1., 1., 1.]);
                normals.push([-1., 1., 1.]);

                normals.push([1., -1., -1.]);
                normals.push([-1., -1., -1.]);
                normals.push([1., 1., -1.]);
                normals.push([-1., 1., -1.]);
            }
        }

        let base_off = vertices.len() as u32;

        vertices.push([0., 0., 0.]);
        vertices.push([-(heightmap.width as f32), 0., 0.]);
        vertices.push([-(heightmap.width as f32), heightmap.height as f32, 0.]);
        vertices.push([0., heightmap.height as f32, 0.]);

        for i in 0..4 {
            uvs.push([0., 0.]);
        }

        normals.push([0., -1., 0.]);
        normals.push([0., -1., 0.]);
        normals.push([0., -1., 0.]);
        normals.push([0., -1., 0.]);

        for normal in &mut normals {

            let len = (normal[0].powf(2.) + normal[1].powf(2.) + normal[2].powf(2.)).sqrt();
            normal[0] /= len;
            normal[1] /= len;
            normal[2] /= len;
        }

        // Compute indices
        let mut indices: Vec<u32> = Vec::new();

        println!("VLEN {}", vertices.len());

        for y in 0..heightmap.height {
            for x in 0..heightmap.width {
                let this_triangle = (heightmap.offset((x, y)) * VERTEX_STRIDE) as u32;

                let mut add = |(x, y, z):( isize, isize, isize), reverse| {
                    let off = |x| ((this_triangle as isize) + x) as u32;
                    let (x, y, z) = (off(x), off(y), off(z));
                    let (x, y, z) = if reverse { (x, z, y) } else { (x, y, z) };
                    indices.push(x);
                    indices.push(y);
                    indices.push(z); 
                };

                // First side, if an edge draw to zero else draw to previous cell so we are
                // manifold
                if x == 0 {
                    add((4, 2, 0), false);
                    add((2, 4, 6), false);
                } else {
                    let last_vertex = -(VERTEX_STRIDE as isize);
                    let going_up = heightmap[(x, y)] <= heightmap[(x - 1, y)];
                    println!("Going up: {}", going_up);
                    add((2, 0, last_vertex + 1), true);
                    add((2, last_vertex + 1, last_vertex + 3), true);
                }

                // Second side (pointing y north), similar shared faces to the x dir
                if y == 0 {
                    add((0, 4, 1), true);
                    add((4, 5, 1), true);
                } else {
                    let going_down = heightmap[(x, y)] <= heightmap[(x, y - 1)];
                    let row_stride: isize = (VERTEX_STRIDE * heightmap.width) as isize;
                    add((0, -row_stride + 2, 1), false);
                    add((-row_stride + 2, -row_stride + 3, 1), false);
                }

                //// Third side
                //// Since the previous cell can share an edge we only draw this at the end
                if x == heightmap.width - 1 {
                    add((7, 1, 3), true);
                    add((7, 5, 1), true);
                }

                //// Fourth side, since the previous cell can share an edge we only draw this at the
                //// end
                if y == heightmap.height - 1 {
                    add((2, 6, 3), false);
                    add((3, 6, 7), false);
                }

                //// Top
                add((2, 1, 0), false);
                add((2, 3, 1), false);
            }
        }

        indices.push(base_off + 2);
        indices.push(base_off + 1);
        indices.push(base_off );

        indices.push(base_off);
        indices.push(base_off + 1);
        indices.push(base_off + 2 );

        indices.push(base_off );
        indices.push(base_off + 3);
        indices.push(base_off + 2);

        indices.push(base_off + 2);
        indices.push(base_off + 3);
        indices.push(base_off );

        let mut result = Model {
            vertices,
            normals,
            uvs,
            indices,
        };
        result.scale((1., 1., heightmap.scale_z));
        result
    }
}
