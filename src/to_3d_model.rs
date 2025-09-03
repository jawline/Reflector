use crate::heightmap::Heightmap;
use log::info;

pub type Point = [f32; 3];
pub type Triangle = [Point; 3];
pub type Quad = [Point; 4];

pub struct Model {
    pub triangles: Vec<Triangle>,
}

fn scale_point(p: &Point, q: &Point) -> Point {
    [p[0] * q[0], p[1] * q[1], p[2] * q[2]]
}

fn scale_triangle(t: &Triangle, q: &Point) -> Triangle {
    [
        scale_point(&t[0], q),
        scale_point(&t[1], q),
        scale_point(&t[2], q),
    ]
}

fn quad_to_triangles(q: &Quad) -> [Triangle; 2] {
    [[q[0], q[1], q[2]], [q[2], q[3], q[0]]]
}

fn row_z_heights(
    heightmap: &Heightmap<f32>,
    (x, y): (usize, usize),
    (xs, ys): (usize, usize),
) -> Vec<f32> {
    let mut result = Vec::new();

    let mut x = y;
    let mut y = y;

    while x < heightmap.width && y < heightmap.height {
        result.push(heightmap[(x, y)]);
        x += xs;
        y += ys;
    }

    return result;
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
        let first_row_z_heights = row_z_heights(&heightmap, (0, 0), (1, 0));
        let last_row_z_heights = row_z_heights(&heightmap, (0, heightmap.height - 1), (1, 0));
        let first_col_z_heights = row_z_heights(&heightmap, (0, 0), (0, 1));
        let last_col_z_heights = row_z_heights(&heightmap, (heightmap.width - 1, 0), (0, 1));
        let extent_z_heights: Vec<f32> = first_row_z_heights
            .iter()
            .chain(last_row_z_heights.iter())
            .chain(first_col_z_heights.iter())
            .chain(last_col_z_heights.iter())
            .cloned()
            .collect();
        let mut quads = Vec::new();

        info!(
            "Meshifying heightmap of size {} {}",
            heightmap.width, heightmap.height
        );

        for y in 0..heightmap.height {
            for x in 0..heightmap.width {
                let z = heightmap[(x, y)];
                let last_z_x = if x == 0 { 0. } else { heightmap[(x - 1, y)] };
                let next_z_x = if x == heightmap.width - 1 {
                    0.
                } else {
                    heightmap[(x + 1, y)]
                };
                let last_z_y = if y == 0 { 0. } else { heightmap[(x, y - 1)] };
                let next_z_y = if y == heightmap.height - 1 {
                    0.
                } else {
                    heightmap[(x, y + 1)]
                };

                let last_diag_1 = if x == 0 || y == 0 {
                    0.
                } else {
                    heightmap[(x - 1, y - 1)]
                };
                let next_diag_1 = if x == heightmap.width - 1 || y == heightmap.height - 1 {
                    0.
                } else {
                    heightmap[(x + 1, y + 1)]
                };

                let last_diag_2 = if x == 0 || y == heightmap.height - 1 {
                    0.
                } else {
                    heightmap[(x - 1, y + 1)]
                };
                let next_diag_2 = if x == heightmap.width - 1 || y == 0 {
                    0.
                } else {
                    heightmap[(x + 1, y - 1)]
                };

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

                let add =
                    |quads: &mut Vec<Quad>, (a, b, c, d): (usize, usize, usize, usize), hz, lz| {
                        let pa = point_to_vertex(a, hz, lz);
                        let pb = point_to_vertex(b, hz, lz);
                        let pc = point_to_vertex(c, hz, lz);
                        let pd = point_to_vertex(d, hz, lz);
                        quads.push([pa, pb, pc, pd]);
                    };

                let compute_left_extent =
                    |quads: &mut Vec<Quad>, first, up_face, down_face, last_z| {
                        let mut heights = Vec::new();

                        heights.push(z);
                        heights.push(last_z);
                        let up = z > last_z;

                        heights.sort_by(f32::total_cmp);

                        let min_z = heights[0];
                        let max_z = heights[1];

                        extent_z_heights.iter().for_each(|&x| {
                            heights.push(x);
                        });
                        heights.push(last_z_x);
                        heights.push(next_z_x);
                        heights.push(last_z_y);
                        heights.push(next_z_y);
                        heights.push(last_diag_1);
                        heights.push(next_diag_1);
                        heights.push(last_diag_2);
                        heights.push(next_diag_2);

                        heights.retain(|&x| x >= min_z && x <= max_z);

                        heights.sort_by(f32::total_cmp);
                        heights.dedup();

                        let face = if up { up_face } else { down_face };

                        for i in 1..heights.len() {
                            add(&mut *quads, face, heights[i], heights[i - 1]);
                        }
                    };

                let compute_right_extent = |quads: &mut Vec<Quad>, face| {
                    let mut heights = Vec::new();

                    heights.push(z);
                    heights.push(0.);

                    extent_z_heights.iter().filter(|&&x| x < z).for_each(|&x| {
                        heights.push(x);
                    });

                    heights.push(last_z_x);
                    heights.push(next_z_x);
                    heights.push(last_z_y);
                    heights.push(next_z_y);

                    heights.sort_by(f32::total_cmp);
                    heights.retain(|&x| x <= z);
                    heights.dedup();

                    for i in 1..heights.len() {
                        add(&mut *quads, face, heights[i], heights[i - 1]);
                    }
                };

                // X face, always draw, reverse order for normal if sloping downward
                if y == 0 || last_z_y != z {
                    let up_face = (4, 5, 1, 0);
                    let down_face = (5, 4, 0, 1);
                    compute_left_extent(&mut quads, y == 0, up_face, down_face, last_z_y);
                }

                // Second side (pointing y north), similar shared faces to the x dir
                if x == 0 || last_z_x != z {
                    let up_face = (0, 3, 7, 4);
                    let down_face = (3, 0, 4, 7);
                    compute_left_extent(&mut quads, x == 0, up_face, down_face, last_z_x);
                }

                // Third side
                // Since the previous cell can share an edge we only draw this at the end
                if y == heightmap.height - 1 {
                    let face = (3, 2, 6, 7);
                    compute_right_extent(&mut quads, face);
                }

                // Fourth side, since the previous cell can share an edge we only draw this at the
                // end
                if x == heightmap.width - 1 {
                    let face = (2, 1, 5, 6);
                    compute_right_extent(&mut quads, face);
                }

                // Top
                add(&mut quads, (0, 1, 2, 3), z, 0.);

                // Base
                add(&mut quads, (5, 4, 7, 6), z, 0.);
            }
        }

        let triangles = quads
            .into_iter()
            .flat_map(|quad| quad_to_triangles(&quad))
            .collect();

        let mut result = Model { triangles };

        result.scale((1., 1., heightmap.scale_z));
        result
    }
}
