use crate::heightmap::Heightmap;
use log::info;

pub type Point = [f32; 3];
pub type TriangleCCW = [Point; 3];

pub struct Model {
    pub triangles: Vec<TriangleCCW>,
}

fn scale_point(p: &Point, q: &Point) -> Point {
    [p[0] * q[0], p[1] * q[1], p[2] * q[2]]
}

fn scale_triangle(t: &TriangleCCW, q: &Point) -> TriangleCCW {
    [
        scale_point(&t[0], q),
        scale_point(&t[1], q),
        scale_point(&t[2], q),
    ]
}

impl Model {
    pub fn scale(&mut self, (scale_x, scale_y, scale_z): (f32, f32, f32)) {
        self.triangles = self
            .triangles
            .iter()
            .map(|x| scale_triangle(x, &[scale_x, scale_y, scale_z]))
            .collect();
    }

    pub fn of_heightmap(heightmap: &Heightmap<f32>) -> Self {
        let mut triangles = Vec::new();

        info!(
            "Meshifying heightmap of size {} {}",
            heightmap.width, heightmap.height
        );

        for y in 0..heightmap.height {
            for x in 0..heightmap.width {
                let z = heightmap[(x, y)];
                let last_z_x = if x == 0 { 0. } else { heightmap[(x - 1, y)] };
                let last_z_y = if y == 0 { 0. } else { heightmap[(x, y - 1)] };

                let point_to_vertex = |pt, z, lz| {
                    let x = x as f32;
                    let y = y as f32;
                    match pt {
                        0 => [x, y, z],
                        1 => [x + 1., y, z],
                        2 => [x + 1., y + 1., z],
                        3 => [x, y + 1., z],
                        4 => [x, y, lz],
                        5 => [x + 1., y, lz],
                        6 => [x + 1., y + 1., lz],
                        7 => [x, y + 1., lz],
                        _ => panic!("impossible point"),
                    }
                };

                let mut add = |(x, y, z): (usize, usize, usize), hz, lz| {
                    let px = point_to_vertex(x, hz, lz);
                    let py = point_to_vertex(y, hz, lz);
                    let pz = point_to_vertex(z, hz, lz);
                    triangles.push([px, py, pz]);
                };

                // X face, always draw, reverse order for normal if sloping downward
                add((1, 0, 5), z, last_z_y);
                add((5, 0, 4), z, last_z_y);

                // Second side (pointing y north), similar shared faces to the x dir
                add((0, 3, 4), z, last_z_x);
                add((4, 3, 7), z, last_z_x);

                // Third side
                // Since the previous cell can share an edge we only draw this at the end
                if y == heightmap.height - 1 {
                    add((7, 3, 2), z, 0.);
                    add((2, 6, 7), z, 0.);
                }

                // Fourth side, since the previous cell can share an edge we only draw this at the
                // end
                if x == heightmap.width - 1 {
                    add((2, 1, 5), z, 0.);
                    add((2, 5, 6), z, 0.);
                }

                // Top
                add((3, 0, 1), z, 0.);
                add((3, 1, 2), z, 0.);
            }
        }

        let base_points = [
            [0., 0., 0.],
            [heightmap.width as f32, 0., 0.],
            [heightmap.width as f32, heightmap.height as f32, 0.],
            [0., heightmap.height as f32, 0.],
        ];
        triangles.push([base_points[1], base_points[0], base_points[2]]);
        triangles.push([base_points[0], base_points[3], base_points[2]]);

        let mut result = Model { triangles };

        result.scale((1., 1., heightmap.scale_z));
        result
    }
}
