use super::types::Model;
use crate::heightmap::Heightmap;
use log::{debug, info};
use std::collections::HashSet;

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

    debug!("{} {} bounds {} {} {} {}", x, y, min_x, min_y, max_x, max_y);

    for px in min_x..=max_x {
        for py in min_y..=max_y {
            debug!("iter {} {}", px, py);
            if px != x || py != y {
                result.push((px, py));
            }
        }
    }

    result
}

fn ascend_spire(previous_spire: &Spire, heightmap: &Heightmap<f32>) -> Vec<Spire> {
    let included_points: HashSet<(usize, usize)> = HashSet::from_iter(
        previous_spire
            .points
            .iter()
            .cloned()
            .filter(|&off| heightmap[off] > previous_spire.end_z),
    );
    let mut handled_points = HashSet::new();
    let mut result_spires = Vec::new();

    for &off in included_points.iter() {
        if !handled_points.contains(&off) {
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

            debug!("Adjacent {:?} {:?}", off, adjacent_points);

            work_list.extend(adjacent_points);

            let mut i = 0;

            while i < work_list.len() {
                let off = work_list[i];

                if included_points.contains(&off) && !handled_points.contains(&off) {
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
        base: false, // Doesn't matter, this isn't an included spire.
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

pub fn of_heightmap(heightmap: &Heightmap<f32>) -> Model {
    info!(
        "Meshifying heightmap of size {} {} {:?}",
        heightmap.width, heightmap.height, heightmap.data
    );

    let spires = discover_spires(heightmap);

    debug!("Discovered spires {:?}", spires);

    unimplemented!()
}
