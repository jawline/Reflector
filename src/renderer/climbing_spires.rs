use super::types::{Model, Point};
use crate::heightmap::Heightmap;
use log::{debug, info};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

pub type Quad = [Point; 4];

#[derive(Debug)]
struct Spire {
    points: HashSet<(usize, usize)>,
    start_z: f32,
    end_z: f32,
    base: bool,
}

fn enumerate_adjacent((x, y): (usize, usize), heightmap: &Heightmap<f32>) -> Vec<(usize, usize)> {
    let mut result = Vec::new();

    let min_x = if x == 0 { x } else { x - 1 };
    let min_y = if y == 0 { y } else { y - 1 };
    let max_x = if x == heightmap.width - 1 { x } else { x + 1 };
    let max_y = if y == heightmap.height - 1 { y } else { y + 1 };

    for px in min_x..=max_x {
        for py in min_y..=max_y {
            if px != x || py != y {
                result.push((px, py));
            }
        }
    }

    result
}

fn ascend_spire(previous_spire: &Spire, heightmap: &Heightmap<f32>) -> Vec<Spire> {
    debug!(
        "scanning for next spire {} {} {}",
        previous_spire.start_z,
        previous_spire.end_z,
        previous_spire.points.len()
    );

    let mut handled_points = HashSet::new();
    let mut result_spires = Vec::new();

    let continues =
        |off| previous_spire.points.contains(&off) && heightmap[off] > previous_spire.end_z;

    for &off in previous_spire.points.iter() {
        if continues(off) && !handled_points.contains(&off) {
            let mut new_spire = Spire {
                points: HashSet::new(),
                start_z: previous_spire.end_z,
                end_z: heightmap[off],
                base: false,
            };

            handled_points.insert(off);
            new_spire.points.insert(off);

            let mut work_list = Vec::new();

            let adjacent_points = enumerate_adjacent(off, heightmap);
            work_list.extend(adjacent_points);

            let mut i = 0;

            while i < work_list.len() {
                let off = work_list[i];

                if continues(off) && !handled_points.contains(&off) {
                    handled_points.insert(off);
                    new_spire.points.insert(off);
                    new_spire.end_z = new_spire.end_z.min(heightmap[off]);
                    work_list.extend(enumerate_adjacent(off, heightmap));
                }

                i += 1;
            }

            result_spires.push(new_spire);
        }
    }

    result_spires
}

fn base_spires(heightmap: &Heightmap<f32>) -> Vec<Spire> {
    // We find the root spires by creating a fake negative spire and then seeing where it joins the
    // real positive shape.
    let mut all_points = HashSet::new();
    for y in 0..heightmap.height {
        for x in 0..heightmap.width {
            all_points.insert((x, y));
        }
    }

    let fake_base = Spire {
        points: all_points,
        start_z: -10000., // Arbitrary, needs to be below the world
        end_z: 0.0,       // Needs to be where the world starts
        base: false,      // Doesn't matter, this isn't an included spire.
    };

    let mut result = ascend_spire(&fake_base, heightmap);

    for spire in &mut result {
        spire.base = true;
    }

    result
}

fn discover_spires(heightmap: &Heightmap<f32>) -> Vec<Spire> {
    let mut worklist = base_spires(heightmap);
    let mut i = 0;

    while i < worklist.len() {
        let spires_above = ascend_spire(&worklist[i], heightmap);

        for spire in spires_above {
            worklist.push(spire);
        }

        i += 1;
    }

    worklist
}

fn draw_spire(quads: &mut Vec<Quad>, spire: &Spire, heightmap: &Heightmap<f32>) {
    for &(x, y) in &spire.points {
        let z = heightmap[(x, y)];

        let point_to_vertex = |pt| {
            let x = x as f32;
            let y = y as f32;
            let ex = 1.;
            let ey = 1.;

            match pt {
                0 => [x, y, spire.end_z],
                1 => [x + ex, y, spire.end_z],
                2 => [x + ex, y + ey, spire.end_z],
                3 => [x, y + ey, spire.end_z],
                4 => [x, y, spire.start_z],
                5 => [x + ex, y, spire.start_z],
                6 => [x + ex, y + ey, spire.start_z],
                7 => [x, y + ey, spire.start_z],
                _ => panic!("impossible point"),
            }
        };

        let add = |quads: &mut Vec<Quad>, (a, b, c, d): (usize, usize, usize, usize)| {
            let pa = point_to_vertex(a);
            let pb = point_to_vertex(b);
            let pc = point_to_vertex(c);
            let pd = point_to_vertex(d);
            quads.push([pa, pb, pc, pd]);
        };

        // Top
        if z <= spire.end_z {
            add(quads, (0, 1, 2, 3));
        }

        // Base
        if spire.base {
            add(quads, (5, 4, 7, 6));
        }

        // X
        if !spire.points.contains(&(x, y - 1)) {
            add(quads, (4, 5, 1, 0));
        }

        // Y
        if !spire.points.contains(&(x - 1, y)) {
            add(quads, (0, 3, 7, 4));
        }

        // X2
        if !spire.points.contains(&(x, y + 1)) {
            add(quads, (3, 2, 6, 7));
        }

        // Y2
        if !spire.points.contains(&(x + 1, y)) {
            add(quads, (2, 1, 5, 6));
        }
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
    info!(
        "Meshifying heightmap of size {} {}",
        heightmap.width, heightmap.height
    );

    let mut heightmap  : Heightmap<f32> = heightmap.clone();

    for point in &mut heightmap.data  {
        *point = (*point * 512.).round() / 512.;
    }

    let heightmap = &heightmap;

    info!("Rounded heightmap");

    // TODO: If we discover the spires on-line while drawing the quads we will use less memory.
    let spires = discover_spires(heightmap);
    info!("Discovered spires {}", spires.len());

    let mut quads: Vec<Quad> = Vec::new();
    for spire in spires {
        draw_spire(&mut quads, &spire, heightmap);
    }

    info!("Rendered to quads");

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
