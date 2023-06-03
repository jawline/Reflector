use itertools::iproduct;
use las::{Read, Reader};
use medians::Medianf64;
use png::Encoder;
use std::env;
use std::ffi::OsStr;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use walkdir::WalkDir;
use log::{debug, info};
            const CONSIDER_NEAREST: usize = 32;

fn flip_y<T: Clone + Copy>(grid_zones: &Vec<T>, (ext_x, ext_y) : (usize, usize)) -> Vec<T> {
    let mut grid_zones_flipped = grid_zones.clone();

    for (grid_x, grid_y) in iproduct!(0..ext_x, 0..ext_y) {
        let offset_output = ((ext_y - grid_y - 1) * ext_x) + grid_x;
        let offset_input = (grid_y * ext_x) + grid_x;
        grid_zones_flipped[offset_output] = grid_zones[offset_input];
    }

    grid_zones_flipped
}

fn fill_with_neighbors(grid_zones: &Vec<Option<f64>>, (ext_x, ext_y) : (usize, usize)) -> Vec<Option<f64>> {
    let mut grid_zones_smoothed = grid_zones.clone();

    for (grid_x, grid_y) in iproduct!(0..ext_x, 0..ext_y) {
        let offset = (grid_y * ext_x) + grid_x;

        if grid_zones[offset].is_none() {

            let nearest_x_start = grid_x.max(CONSIDER_NEAREST / 2) - (CONSIDER_NEAREST / 2);
            let nearest_y_start = grid_y.max(CONSIDER_NEAREST / 2) - (CONSIDER_NEAREST / 2);

            let mut nearest = Vec::new();

            for (near_x, near_y) in iproduct!(
                nearest_x_start..(nearest_x_start + CONSIDER_NEAREST).min(ext_x),
                nearest_y_start..(nearest_y_start + CONSIDER_NEAREST).min(ext_y)
            ) {
                //println!("Consider {} {} for gapfill of {} {}", near_x, near_y, grid_x, grid_y);
                let offset = (near_y * ext_x) + near_x;
                if let Some(mode) = grid_zones[offset] {
                    nearest.push(mode);
                }
            }

            if nearest.len() > 0 {
                grid_zones_smoothed[offset] = Some(nearest.median().unwrap());
            } 
        }
    }

    grid_zones_smoothed
}


#[derive(Debug, Clone)]
struct GridZone {
    pub points: Vec<f64>,
}

fn main() {
    env_logger::init();
    let path = env::args().nth(1).unwrap();
    println!("Computing path: {}", path);

    let mut max_x = None;
    let mut min_x = None;
    let mut max_z = None;
    let mut min_z = None;
    let mut max_y = None;
    let mut min_y = None;

    println!("Beginning first pass");

    for entry in WalkDir::new(path.clone())
        .max_depth(100)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| e.path().extension() == Some(&OsStr::new("las")))
    {
        debug!("Loading path: {:?}", entry.path());
        let mut reader = Reader::from_path(entry.path()).expect("Unable to open reader");

        for wrapped_point in reader.points() {
            let wrapped_point = wrapped_point.unwrap();

            max_x = Some(max_x.unwrap_or(wrapped_point.x).max(wrapped_point.x));
            max_y = Some(max_y.unwrap_or(wrapped_point.y).max(wrapped_point.y));
            max_z = Some(max_z.unwrap_or(wrapped_point.z).max(wrapped_point.z));
            min_x = Some(min_x.unwrap_or(wrapped_point.x).min(wrapped_point.x));
            min_y = Some(min_y.unwrap_or(wrapped_point.y).min(wrapped_point.y));
            min_z = Some(min_z.unwrap_or(wrapped_point.z).min(wrapped_point.z));
        }
    }

    let (min_x, max_x, min_y, max_y, min_z, max_z) = (
        min_x.unwrap(),
        max_x.unwrap(),
        min_y.unwrap(),
        max_y.unwrap(),
        min_z.unwrap(),
        max_z.unwrap(),
    );

    info!(
        "Bounds: {} {} {} {} {} {}",
        min_x, max_x, min_y, max_y, min_z, max_z
    );

    println!("Main pass, summarizing grid squares");

    let dpx = 2.;

    let GRID_X = (((max_x - min_x) * dpx).ceil() as usize);
    let GRID_Y = (((max_y - min_y) * dpx).ceil() as usize);
    let EXT_X = GRID_X + 1;
    let EXT_Y = GRID_Y + 1;

    info!("Derived GRID_X: {}, Derived GRID_Y: {}", GRID_X, GRID_Y);

    let mut grid_zones = Vec::new();
    grid_zones.resize(EXT_X * EXT_Y, GridZone { points: Vec::new() });

    for entry in WalkDir::new(path)
        .max_depth(100)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| e.path().extension() == Some(&OsStr::new("las")))
    {
        debug!("Loading path: {:?}", entry.path());
        let mut reader = Reader::from_path(entry.path()).expect("Unable to open reader");
        for wrapped_point in reader.points() {
            let wrapped_point = wrapped_point.unwrap();
            let x_ratio = ((wrapped_point.x - min_x) / (max_x - min_x));
            let y_ratio = ((wrapped_point.y - min_y) / (max_y - min_y));
            //println!("{} {}", x_ratio, z_ratio);
            let grid_x = (x_ratio * GRID_X as f64).floor() as usize;
            let grid_y = (y_ratio * GRID_Y as f64).floor() as usize;
            //println!("{} {}", grid_x, grid_z);

            let zone = &mut grid_zones[(grid_y * EXT_X) + grid_x];
            zone.points.push(wrapped_point.z);
        }
    }

    let grid_zones: Vec<Option<f64>> = grid_zones
        .iter()
        .map(|grid_zone| {
            if grid_zone.points.is_empty() {
                None
            } else {
                Some(grid_zone.points.median().unwrap())
            }
        })
        .collect();

    info!("Flipping the Y axis");
    let mut grid_zones = flip_y(&grid_zones, (EXT_X, EXT_Y));

    for i in 0..16 {
        info!("Neighbor filling round {}", i);
        grid_zones = fill_with_neighbors(&grid_zones, (EXT_X, EXT_Y))
    }

    // Here every point will be some
    info!("Normalizing Z axis");
    let grid_zones: Vec<f64> = grid_zones.iter().map(|x| x.unwrap_or(0.) / max_z).collect();

    info!("Writing to file");

    let path = Path::new(r"./heatmap.png");
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, (EXT_X) as u32, (EXT_Y) as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    let zones_as_bytes: Vec<u8> = grid_zones.iter().map(|x| ((1. - x) * 255.) as u8).collect();

    writer.write_image_data(&zones_as_bytes).unwrap(); // Save
}
