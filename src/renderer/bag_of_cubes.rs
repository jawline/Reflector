use super::types::{Model, Point};
use crate::heightmap::Heightmap;
use log::info;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

pub type Quad = [Point; 4];

fn find_collateable_row(
    heightmap: &Heightmap<f32>,
    done: &HashSet<(usize, usize)>,
    z: f32,
    (x, y): (usize, usize),
    dirswap: bool,
) -> usize {
    let delta = 0.001;
    let start = if dirswap { y } else { x };
    let mut end = if dirswap { y } else { x };

    let stop_condition = if dirswap {
        heightmap.height - 1
    } else {
        heightmap.width - 1
    };

    while end < stop_condition {
        let lx = end + 1;
        let index = if dirswap { (x, lx) } else { (lx, y) };
        if !done.contains(&index) && (heightmap[index] - z).abs() < delta {
            end += 1;
        } else {
            break;
        }
    }

    return end - start;
}

fn find_rectangle(
    heightmap: &Heightmap<f32>,
    done: &HashSet<(usize, usize)>,
    z: f32,
    (origin_x, origin_y): (usize, usize),
    dirswap: bool,
) -> (usize, usize) {
    let mut thresh_x = 0;
    let mut thresh_y = 0;

    if dirswap {
        thresh_y = find_collateable_row(heightmap, done, z, (origin_x, origin_y), true);
    } else {
        thresh_x = find_collateable_row(heightmap, done, z, (origin_x, origin_y), false);
    }

    let incr = |x: &mut usize, y: &mut usize| {
        if dirswap {
            *x += 1;
        } else {
            *y += 1;
        }
    };

    let found_anything = thresh_x > 1 || thresh_y > 1;

    if found_anything {
        loop {
            let cx = origin_x + thresh_x;
            let cy = origin_y + thresh_y;

            if cx >= heightmap.width - 1 || cy >= heightmap.height - 1 {
                break;
            }

            let index = if dirswap {
                (origin_x + thresh_x + 1, origin_y)
            } else {
                (origin_x, origin_y + thresh_y + 1)
            };

            let row = find_collateable_row(heightmap, &done, z, index, dirswap);
            let thresh = if dirswap { thresh_y } else { thresh_x };

            if row < thresh {
                break;
            }

            incr(&mut thresh_x, &mut thresh_y);
        }
    }

    (thresh_x, thresh_y)
}

fn find_largest_rectangle(
    heightmap: &Heightmap<f32>,
    done: &HashSet<(usize, usize)>,
    z: f32,
    p: (usize, usize),
) -> (usize, usize) {
    let (r1x, r1y) = find_rectangle(heightmap, done, z, p, false);
    let (r2x, r2y) = find_rectangle(heightmap, done, z, p, true);

    if (r1x * r1y) > (r2x * r2y) {
        (p.0 + r1x, p.1 + r1y)
    } else {
        (p.0 + r2x, p.1 + r2y)
    }
}

// An imprecise hash we use for vertex reduction.
#[derive(Debug, Copy, Clone)]
struct HashP(Point);
fn ceq(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}
impl PartialEq for HashP {
    fn eq(&self, other: &Self) -> bool {
        ceq(self.0[0], other.0[0]) && ceq(self.0[1], other.0[1]) && ceq(self.0[2], other.0[2])
    }
}

impl Eq for HashP {}

fn hashf(f: f32) -> u32 {
    if f.is_nan() {
        0
    } else {
        f.to_bits()
    }
}

impl Hash for HashP {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(hashf(self.0[0]) + hashf(self.0[1]) + hashf(self.0[1]));
    }
}

pub fn of_heightmap(heightmap: &Heightmap<f32>) -> Model {
    let mut done: HashSet<(usize, usize)> = HashSet::new();
    let mut quads = Vec::new();

    info!(
        "Meshifying heightmap of size {} {}",
        heightmap.width, heightmap.height
    );

    for y in 0..heightmap.height {
        for x in 0..heightmap.width {
            if done.contains(&(x, y)) {
                continue;
            }
            let z = heightmap[(x, y)];

            let (end_x, end_y) = find_largest_rectangle(heightmap, &done, z, (x, y));

            if x != end_x || y != end_y {}

            for y in y..=end_y {
                for x in x..=end_x {
                    done.insert((x, y));
                }
            }

            let point_to_vertex = |pt, z, lz| {
                let x = x as f32;
                let y = y as f32;
                let end_x = end_x as f32;
                let end_y = end_y as f32;
                let ex = 1. + (end_x - x);
                let ey = 1. + (end_y - y);

                match pt {
                    0 => [x, y, z],
                    1 => [x + ex, y, z],
                    2 => [x + ex, y + ey, z],
                    3 => [x, y + ey, z],
                    4 => [x, y, lz],
                    5 => [x + ex, y, lz],
                    6 => [x + ex, y + ey, lz],
                    7 => [x, y + ey, lz],
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

            // Top
            add(&mut quads, (0, 1, 2, 3), z, 0.);

            // Base
            add(&mut quads, (5, 4, 7, 6), z, 0.);

            // X
            add(&mut quads, (4, 5, 1, 0), z, 0.);

            // Y
            add(&mut quads, (0, 3, 7, 4), z, 0.);

            // X2
            add(&mut quads, (3, 2, 6, 7), z, 0.);

            // Y2
            add(&mut quads, (2, 1, 5, 6), z, 0.);
        }
    }

    info!("Producing triangles");

    let mut triangle_indices = HashMap::new();

    let mut vertices = Vec::new();
    let mut triangles = Vec::new();

    let mut add_vertex = |point| match triangle_indices.get(&HashP(point)) {
        Some(x) => *x,
        None => {
            let idx = vertices.len();
            triangle_indices.insert(HashP(point), idx);
            vertices.push(point);
            idx
        }
    };

    let mut quad_to_triangles = |q: &Quad| {
        let q0 = add_vertex(q[0]);
        let q1 = add_vertex(q[1]);
        let q2 = add_vertex(q[2]);
        let q3 = add_vertex(q[3]);
        triangles.push([q0, q1, q2]);
        triangles.push([q2, q3, q0]);
    };

    info!("Processing {} quads", quads.len());

    quads.into_iter().for_each(|quad| quad_to_triangles(&quad));

    let mut result = Model {
        vertices,
        triangles,
    };

    info!("Scaling");

    result.scale((1., 1., heightmap.scale_z));
    result
}
