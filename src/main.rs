mod las_data;
use crate::las_data::LasData;
use clap::Parser;
use itertools::iproduct;
use log::info;
use medians::Medianf64;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const CONSIDER_NEAREST: usize = 32;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    las_folder_path: String,

    #[arg(short, long)]
    output_path: String,

    #[arg(short, long, default_value_t = 1)]
    rounds_of_interpolated_hole_filling: usize,
}

struct Heightmap<T: Clone + Copy> {
    data: Vec<T>,
    width: usize,
    height: usize,
}

impl<T: Clone + Copy> Heightmap<T> {
    fn flip_y(&self) -> Self {
        let mut flipped = self.data.clone();

        for (grid_x, grid_y) in iproduct!(0..self.width, 0..self.height) {
            let offset_output = ((self.height - grid_y - 1) * self.width) + grid_x;
            let offset_input = (grid_y * self.width) + grid_x;
            flipped[offset_output] = flipped[offset_input];
        }

        Self {
            data: flipped,
            width: self.width,
            height: self.height,
        }
    }
}

impl Heightmap<Option<f64>> {
    fn interpolate_missing_using_neighbors(&self) -> Self {
        let mut grid_zones_smoothed = self.data.clone();

        for (grid_x, grid_y) in iproduct!(0..self.width, 0..self.height) {
            let offset = (grid_y * self.width) + grid_x;

            if self.data[offset].is_none() {
                let nearest_x_start = grid_x.max(CONSIDER_NEAREST / 2) - (CONSIDER_NEAREST / 2);
                let nearest_y_start = grid_y.max(CONSIDER_NEAREST / 2) - (CONSIDER_NEAREST / 2);

                let mut nearest = Vec::new();

                for (near_x, near_y) in iproduct!(
                    nearest_x_start..(nearest_x_start + CONSIDER_NEAREST).min(self.width),
                    nearest_y_start..(nearest_y_start + CONSIDER_NEAREST).min(self.height)
                ) {
                    //println!("Consider {} {} for gapfill of {} {}", near_x, near_y, grid_x, grid_y);
                    let offset = (near_y * self.width) + near_x;
                    if let Some(mode) = self.data[offset] {
                        nearest.push(mode);
                    }
                }

                if nearest.len() > 0 {
                    grid_zones_smoothed[offset] = Some(nearest.median().unwrap());
                }
            }
        }

        Self {
            data: grid_zones_smoothed,
            width: self.width,
            height: self.height,
        }
    }

    fn normalize_z_by_and_fill_none_with_zero(&self, max_z: f64) -> Heightmap<f64> {
        Heightmap {
            data: self.data.iter().map(|x| x.unwrap_or(0.) / max_z).collect(),
            width: self.width,
            height: self.height,
        }
    }
}

fn las_data_to_opt_height_map(
    data: &LasData,
    pixels_per_distance_unit: f64,
) -> Heightmap<Option<f64>> {
    let grid_x = ((data.max_x - data.min_x) * pixels_per_distance_unit).ceil() as usize;
    let grid_y = ((data.max_y - data.min_y) * pixels_per_distance_unit).ceil() as usize;
    let ext_x = grid_x + 1;
    let ext_y = grid_y + 1;

    info!("Derived GRID_X: {}, Derived GRID_Y: {}", grid_x, grid_y);

    let mut grid_zones = Vec::new();
    grid_zones.resize(ext_x * ext_y, GridZone { points: Vec::new() });

    for (px, py, pz) in &data.points {
        let x_ratio = (px - data.min_x) / (data.max_x - data.min_x);
        let y_ratio = (py - data.min_y) / (data.max_y - data.min_y);
        let grid_x = (x_ratio * grid_x as f64).floor() as usize;
        let grid_y = (y_ratio * grid_y as f64).floor() as usize;

        let zone = &mut grid_zones[(grid_y * ext_x) + grid_x];
        zone.points.push(*pz);
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

    Heightmap {
        data: grid_zones,
        width: ext_x,
        height: ext_y,
    }
}

#[derive(Debug, Clone)]
struct GridZone {
    pub points: Vec<f64>,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    println!("Reading LAS files from: {}", args.las_folder_path);

    let data = LasData::load_from_directory(&args.las_folder_path);

    info!(
        "Bounds: {} {} {} {} {} {}",
        data.min_x, data.max_x, data.min_y, data.max_y, data.min_z, data.max_z
    );

    println!("Main pass, summarizing grid squares");

    let grid_zones = las_data_to_opt_height_map(&data, 1.);

    info!("Flipping the Y axis");
    let mut grid_zones = grid_zones.flip_y();

    for i in 0..args.rounds_of_interpolated_hole_filling {
        info!("Neighbor filling round {}", i);
        grid_zones = grid_zones.interpolate_missing_using_neighbors();
    }

    // Here every point will be some
    info!("Normalizing Z axis");
    let grid_zones = grid_zones.normalize_z_by_and_fill_none_with_zero(data.max_z);

    info!("Writing to file");

    let path = Path::new(r"./heatmap.png");
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, (grid_zones.width) as u32, (grid_zones.height) as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    let zones_as_bytes: Vec<u8> = grid_zones
        .data
        .iter()
        .map(|x| ((1. - x) * 255.) as u8)
        .collect();

    writer.write_image_data(&zones_as_bytes).unwrap(); // Save
}
