use crate::las_data::Limits;
use bitvector::BitVector;
use fastblur::gaussian_blur_asymmetric_single_channel;
use itertools::iproduct;
use log::info;
use quantiles::ckms::CKMS;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::fs::File;
use std::io::{BufWriter, Error, Write};
use std::ops::{Index, IndexMut};
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
        let idx = self.offset((x, y));
        &self.data[idx]
    }
}

impl<T: Clone + Copy> IndexMut<(usize, usize)> for Heightmap<T> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        let idx = self.offset((x, y));
        &mut self.data[idx]
    }
}

pub enum InterpolationMode {
    Min,
    Max,
    Mean,
    Percentile(f32),
}

pub fn mean(neighbors: &[f32]) -> Option<f32> {
    let len = neighbors.len();
    if len > 0 {
        Some(neighbors.iter().sum::<f32>() / (len as f32))
    } else {
        None
    }
}

pub fn median(neighbors: &[f32], percentile: f32) -> Option<f32> {
    let mut neighbors: Vec<f32> = neighbors.to_vec();
    neighbors.sort_by(|a, b| a.total_cmp(b));
    if neighbors.len() > 0 {
        let grid_elt = (neighbors.len() as f32 * percentile) as usize;
        Some(neighbors[grid_elt])
    } else {
        None
    }
}

pub fn min_elt(neighbors: &[f32]) -> Option<f32> {
    neighbors.iter().min_by(|a, b| a.total_cmp(b)).copied()
}

pub fn max_elt(neighbors: &[f32]) -> Option<f32> {
    neighbors.iter().max_by(|a, b| a.total_cmp(b)).copied()
}

#[derive(Debug)]
struct VoidAndPerimeter {
    void: Vec<(usize, usize)>,
    perimeter: Vec<(usize, usize)>,
}

impl<T: Copy + Clone> Heightmap<Option<T>> {
    pub fn proportion_of_empty_cells(&self) -> f64 {
        self.data.iter().filter(|x| x.is_none()).count() as f64 / self.data.len() as f64
    }

    fn expand_void(&self, x: usize, y: usize) -> VoidAndPerimeter {
        let mut void = Vec::new();
        let mut perimeter = Vec::new();

        let mut seen = HashSet::new();
        let mut worklist = VecDeque::new();

        macro_rules! add {
            ($l:expr) => {
                match seen.contains(&$l) {
                    true => (),
                    false => {
                        seen.insert($l);
                        worklist.push_back($l);
                    }
                }
            };
        }

        add!((x, y));

        while let Some((x, y)) = worklist.pop_front() {
            match self[(x, y)] {
                Some(_) => perimeter.push((x, y)),
                None =>
                /* Void */
                {
                    void.push((x, y));
                    let can_go_west = x > 0;
                    let can_go_east = x < self.width - 1;
                    let can_go_north = y > 0;
                    let can_go_south = y < self.height - 1;

                    if can_go_west {
                        add!((x - 1, y));
                    }

                    if can_go_east {
                        add!((x + 1, y));
                    }

                    if can_go_north {
                        add!((x, y - 1));
                    }

                    if can_go_south {
                        add!((x, y + 1));
                    }
                }
            }
        }

        VoidAndPerimeter { perimeter, void }
    }
}

impl Heightmap<Option<f32>> {
    pub fn flood_fill(&mut self) {
        let mut seen = BitVector::new(self.width * self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                let is_void = self[(x, y)].is_none();
                let considered = seen.contains(self.offset((x, y)));
                if is_void && !considered {
                    let total_void = self.expand_void(x, y);

                    let perimeter_points: Vec<f32> = total_void
                        .perimeter
                        .iter()
                        .map(|&(x, y)| {
                            let pt: Option<f32> = self[(x, y)];
                            pt.unwrap()
                        })
                        .collect();

                    for &(x, y) in &total_void.void {
                        seen.insert(self.offset((x, y)));
                    }

                    if total_void.void.len() > (self.width * self.height) / 200 {
                        info!("Large void {}", total_void.void.len());

                        let med = median(&perimeter_points, 0.1);

                        if let Some(med) = med {
                            for (x, y) in total_void.void {
                                self[(x, y)] = Some(med);
                            }
                        }
                    }
                }
            }
        }
    }

    fn cast_ray(
        &self,
        (input_x, input_y): (usize, usize),
        x_mod: usize,
        y_mod: usize,
        limit: usize,
    ) -> Option<(usize, usize)> {
        for i in 0..(limit) {
            let x = input_x + (x_mod * i);
            let y = input_y + (y_mod * i);

            if x < self.width && y < self.height && self[(x, y)].is_some() {
                return Some((x, y));
            }
        }
        None
    }

    fn fill_ray(
        &mut self,
        (start_x, start_y): (usize, usize),
        (end_x, end_y): (usize, usize),

        x_mod: usize,
        y_mod: usize,
    ) {
        let val_start = self[(start_x, start_y)].unwrap();
        let val_end = self[(end_x, end_y)].unwrap();
        let med = (val_end + val_start) / 2.;

        let mut cur_x = start_x + x_mod;
        let mut cur_y = start_y + y_mod;

        while (cur_x != end_x) || (cur_y != end_y) {
            self[(cur_x, cur_y)] = Some(med);
            cur_x += x_mod;
            cur_y += y_mod;
        }
    }

    pub fn building_ray_filler(&mut self, max_distance: usize) {
        let min_size_required_to_fill_a_ray = 3;

        for y in 0..(self.height - min_size_required_to_fill_a_ray) {
            for x in 0..(self.width - min_size_required_to_fill_a_ray) {
                let is_some = self[(x, y)].is_some();

                // We only check in 2d as we're doing this over the entire image in this direction
                // so the behind is already checked.
                let is_cliff_x = self[(x + 1, y)].is_none();
                let is_cliff_y = self[(x, y + 1)].is_none();

                if is_some && is_cliff_x {
                    match self.cast_ray((x + 1, y), 1, 0, max_distance) {
                        Some((ray_x, ray_y)) => {
                            self.fill_ray((x, y), (ray_x, ray_y), 1, 0);
                        }
                        None => (),
                    }
                }

                if is_some && is_cliff_y {
                    match self.cast_ray((x, y + 1), 0, 1, max_distance) {
                        Some((ray_x, ray_y)) => {
                            self.fill_ray((x, y), (ray_x, ray_y), 0, 1);
                        }
                        None => (),
                    }
                }
            }
        }
    }

    pub fn interpolate_missing_using_neighbors(
        &self,
        mode: InterpolationMode,
        consider_nearest: usize,
    ) -> Self {
        let mut grid_zones_smoothed = Vec::with_capacity(self.width * self.height);

        // We do not use itertools here as this code needs to push to grid_zones_smoothed in a
        // specific order.
        for grid_y in 0..self.height {
            for grid_x in 0..self.width {
                let offset = (grid_y * self.width) + grid_x;

                let slot = match self.data[offset] {
                    Some(data) => Some(data),
                    None => {
                        let mut neighbors = Vec::with_capacity(consider_nearest * consider_nearest);

                        let nearest_x_start = if grid_x < consider_nearest {
                            0
                        } else {
                            grid_x - consider_nearest
                        };

                        let nearest_y_start = if grid_y < consider_nearest {
                            0
                        } else {
                            grid_y - consider_nearest
                        };

                        let nearest_x_end = (grid_x + consider_nearest).min(self.width);
                        let nearest_y_end = (grid_y + consider_nearest).min(self.height);

                        for (near_x, near_y) in iproduct!(
                            nearest_x_start..nearest_x_end,
                            nearest_y_start..nearest_y_end
                        ) {
                            let offset = (near_y * self.width) + near_x;
                            if let Some(mode) = self.data[offset] {
                                neighbors.push(mode);
                            }
                        }

                        // TODO: If we make consider_nearest make sense, fix this * 2
                        // Only fill in nodes for which we have a reasonable amount of nearby data.
                        let expected_neighbors = (consider_nearest * 2) * (consider_nearest * 2);
                        if neighbors.len() < (expected_neighbors / 4) {
                            None
                        } else {
                            match mode {
                                InterpolationMode::Min => min_elt(&neighbors),
                                InterpolationMode::Max => max_elt(&neighbors),
                                InterpolationMode::Mean => mean(&neighbors),
                                InterpolationMode::Percentile(flt) => median(&neighbors, flt),
                            }
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
        self.data
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
        gaussian_blur_asymmetric_single_channel(&mut self.data, self.width, self.height, 0.1, 0.1);
    }

    pub fn to_f32(&self) -> Heightmap<f32> {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|x| (*x as f32 / 255.) * self.scale_z)
            .collect();
        Heightmap {
            data,
            width: self.width,
            height: self.height,
            pixels_per_distance_unit: self.pixels_per_distance_unit,
            scale_z: 1.,
        }
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
        grid_zones.resize_with(ext_x * ext_y, || CKMS::new(0.1));
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
