use crate::las_data::Limits;
use fastblur::gaussian_blur_asymmetric_single_channel;
use itertools::iproduct;
use log::info;
use quantiles::ckms::CKMS;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Error, Write};
use std::ops::Index;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct Heightmap<T: Clone + Copy> {
    pub data: Vec<T>,
    pub width: usize,
    pub height: usize,
    pub scale_z: f32,

    // We keep track of this so we can scale the stl or 3D models
    pub pixels_per_distance_unit: f32,
}

impl<T: Clone + Copy> Heightmap<T> {
    pub fn flip_y(&self) -> Self {
        let mut flipped = self.data.clone();

        for (grid_x, grid_y) in iproduct!(0..self.width, 0..self.height) {
            let offset_output = ((self.height - grid_y - 1) * self.width) + grid_x;
            let offset_input = (grid_y * self.width) + grid_x;
            flipped[offset_output] = self.data[offset_input];
        }

        Self {
            data: flipped,
            width: self.width,
            height: self.height,
            pixels_per_distance_unit: self.pixels_per_distance_unit,
            scale_z: self.scale_z,
        }
    }

    pub fn offset(&self, (x, y): (usize, usize)) -> usize {
        (y * self.width) + x
    }
}

impl<T: Clone + Copy> Index<(usize, usize)> for Heightmap<T> {
    type Output = T;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        let idx = (y * self.width) + x;
        &self.data[idx]
    }
}

impl Heightmap<Option<f32>> {
    pub fn interpolate_missing_using_neighbors(&self, consider_nearest: usize) -> Self {
        let mut grid_zones_smoothed = Vec::with_capacity(self.width * self.height);

        // We do not use itertools here as this code needs to push to grid_zones_smoothed in a
        // specific order.
        for grid_y in 0..self.height {
            for grid_x in 0..self.width {
                let offset = (grid_y * self.width) + grid_x;

                let slot = match self.data[offset] {
                    Some(data) => Some(data),
                    None => {
                        let mut nearest = Vec::with_capacity(consider_nearest * consider_nearest);

                        let nearest_x_start = grid_x.max(consider_nearest) - (consider_nearest);
                        let nearest_y_start = grid_y.max(consider_nearest) - (consider_nearest);

                        for (near_x, near_y) in iproduct!(
                            nearest_x_start..(nearest_x_start + consider_nearest).min(self.width),
                            nearest_y_start..(nearest_y_start + consider_nearest).min(self.height)
                        ) {
                            let offset = (near_y * self.width) + near_x;
                            if let Some(mode) = self.data[offset] {
                                nearest.push(mode);
                            }
                        }

                        if nearest.len() > 0 {
                            Some(*nearest.iter().min_by(|a, b| a.total_cmp(b)).unwrap())
                        } else {
                            None
                        }
                    }
                };
                grid_zones_smoothed.push(slot);
            }
        }

        Self {
            data: grid_zones_smoothed,
            width: self.width,
            height: self.height,
            pixels_per_distance_unit: self.pixels_per_distance_unit,
            scale_z: self.scale_z,
        }
    }

    pub fn fill_none_with_zero_and_add_base(
        &self,
        base_height: f32,
        base_height_when_none: f32,
    ) -> Heightmap<f32> {
        Heightmap {
            data: self
                .data
                .iter()
                .map(|x| match x {
                    Some(x) => x + base_height,
                    None => base_height_when_none,
                })
                .collect(),
            width: self.width,
            height: self.height,
            pixels_per_distance_unit: self.pixels_per_distance_unit,
            scale_z: self.scale_z,
        }
    }

    pub fn max_z(&self) -> f32 {
        self
            .data
            .iter()
            .filter_map(|x| *x)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    }

    pub fn normalize_z_by(&self, max_z: f32) -> Self {
        Heightmap {
            data: self
                .data
                .iter()
                .map(|x| match x {
                    Some(x) => Some(x / max_z),
                    None => None,
                })
                .collect(),
            width: self.width,
            height: self.height,
            pixels_per_distance_unit: self.pixels_per_distance_unit,
            scale_z: max_z,
        }
    }

    pub fn serialize<W>(self, mut to: W) -> Result<(), Error>
    where
        W: Write,
    {
        let width = self.width as u16;
        let height = self.height as u16;
        let scale_z = self.scale_z;
        println!("{} {} {}", width, height, scale_z);

        to.write(&width.to_le_bytes())?;
        to.write(&height.to_le_bytes())?;
        to.write(&scale_z.to_le_bytes())?;

        for point in self.data {
            let point = match point {
                Some(x) => x,
                None => f32::NAN,
            };
            to.write(&point.to_le_bytes())?;
        }

        Ok(())
    }
}

impl Heightmap<u8> {
    pub fn blur(&mut self) {
        gaussian_blur_asymmetric_single_channel(
            &mut self.data,
            self.width,
            self.height,
            0.25,
            0.25,
        );
    }

    pub fn write_to_png(&self, path: &str) {
        let path = Path::new(path);
        let file = File::create(path).unwrap();
        let ref mut w = BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, (self.width) as u32, (self.height) as u32);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();

        writer.write_image_data(&self.data).unwrap(); // Save
    }
}

impl Heightmap<f32> {
    pub fn to_u8(&self, max_y_is_low: bool) -> Heightmap<u8> {
        let data: Vec<u8> = self
            .data
            .iter()
            .map(|x| ((if max_y_is_low { 1. - x } else { *x }) * 255.) as u8)
            .collect();
        Heightmap {
            data,
            width: self.width,
            height: self.height,
            pixels_per_distance_unit: self.pixels_per_distance_unit,
            scale_z: 1.,
        }
    }

    pub fn add_base(&self, depth: f32) -> Self {
        Heightmap {
            data: self.data.iter().map(|x| x + depth).collect(),
            width: self.width,
            height: self.height,
            pixels_per_distance_unit: self.pixels_per_distance_unit,
            scale_z: self.scale_z,
        }
    }

    pub fn max_z(&self) -> f32 {
        *self.data.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
    }

    pub fn normalize_z_by(&self, max_z: f32) -> Self {
        Heightmap {
            data: self.data.iter().map(|x| x / max_z).collect(),
            width: self.width,
            height: self.height,
            pixels_per_distance_unit: self.pixels_per_distance_unit,
            scale_z: max_z,
        }
    }

    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Self {
        Heightmap {
            data: self.data.iter().map(|x| f(*x)).collect(),
            width: self.width,
            height: self.height,
            pixels_per_distance_unit: self.pixels_per_distance_unit,
            scale_z: self.scale_z,
        }
    }
}

pub struct StreamingHeightmap {
    grid_zones: Vec<CKMS<f32>>,
    grid_x: usize,
    grid_y: usize,
    ext_x: usize,
    ext_y: usize,
    limits: Limits,
    pixels_per_distance_unit: f32,
}

impl StreamingHeightmap {
    pub fn new(limits: &Limits, pixels_per_distance_unit: f32) -> Self {
        let grid_x = ((limits.max_x - limits.min_x) * pixels_per_distance_unit).ceil() as usize;
        let grid_y = ((limits.max_y - limits.min_y) * pixels_per_distance_unit).ceil() as usize;
        info!("Derived GRID_X: {}, Derived GRID_Y: {}", grid_x, grid_y);

        let ext_x = grid_x + 1;
        let ext_y = grid_y + 1;
        let mut grid_zones = Vec::new();
        grid_zones.resize_with(ext_x * ext_y, || CKMS::new(0.02));
        Self {
            grid_x,
            grid_y,
            ext_x,
            ext_y,
            limits: limits.clone(),
            grid_zones,
            pixels_per_distance_unit,
        }
    }

    pub fn add(&mut self, (px, py, pz): (f32, f32, f32)) {
        let x_ratio = (px - self.limits.min_x) / (self.limits.max_x - self.limits.min_x);
        let y_ratio = (py - self.limits.min_y) / (self.limits.max_y - self.limits.min_y);
        let grid_x = (x_ratio * self.grid_x as f32).floor() as usize;
        let grid_y = (y_ratio * self.grid_y as f32).floor() as usize;
        let zone = &mut self.grid_zones[(grid_y * self.ext_x) + grid_x];
        zone.insert(pz);
    }

    pub fn finalize(&self) -> Heightmap<Option<f32>> {
        info!("Constructed quantograms");

        let grid_zones: Vec<Option<f32>> = self
            .grid_zones
            .iter()
            .map(|grid_zone| grid_zone.query(0.5).map(|(_c, t)| t))
            .collect();

        info!("Summarized grid zones");

        Heightmap {
            data: grid_zones,
            width: self.ext_x,
            height: self.ext_y,
            pixels_per_distance_unit: self.pixels_per_distance_unit,
            scale_z: 1.,
        }
    }
}
