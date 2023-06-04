use crate::las_data::LasData;
use itertools::iproduct;
use log::info;
use medians::Medianf64;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

pub struct Heightmap<T: Clone + Copy> {
    pub data: Vec<T>,
    pub width: usize,
    pub height: usize,
}

impl<T: Clone + Copy> Heightmap<T> {
    pub fn flip_y(&self) -> Self {
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
    pub fn interpolate_missing_using_neighbors(&self, consider_nearest: usize) -> Self {
        let mut grid_zones_smoothed = self.data.clone();

        for (grid_x, grid_y) in iproduct!(0..self.width, 0..self.height) {
            let offset = (grid_y * self.width) + grid_x;

            if self.data[offset].is_none() {
                let nearest_x_start = grid_x.max(consider_nearest / 2) - (consider_nearest / 2);
                let nearest_y_start = grid_y.max(consider_nearest / 2) - (consider_nearest / 2);

                let mut nearest = Vec::new();

                for (near_x, near_y) in iproduct!(
                    nearest_x_start..(nearest_x_start + consider_nearest).min(self.width),
                    nearest_y_start..(nearest_y_start + consider_nearest).min(self.height)
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

    pub fn normalize_z_by_and_fill_none_with_zero(&self, max_z: f64) -> Heightmap<f64> {
        Heightmap {
            data: self.data.iter().map(|x| x.unwrap_or(0.) / max_z).collect(),
            width: self.width,
            height: self.height,
        }
    }
}

impl Heightmap<f64> {
    pub fn write_to_png(&self, path: &str) {
        let path = Path::new(path);
        let file = File::create(path).unwrap();
        let ref mut w = BufWriter::new(file);
        let mut encoder =
            png::Encoder::new(w, (self.width) as u32, (self.height) as u32);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        let zones_as_bytes: Vec<u8> = self
            .data
            .iter()
            .map(|x| ((1. - x) * 255.) as u8)
            .collect();

        writer.write_image_data(&zones_as_bytes).unwrap(); // Save
    }
}

#[derive(Debug, Clone)]
struct GridZone {
    pub points: Vec<f64>,
}

pub fn las_data_to_opt_height_map(
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
