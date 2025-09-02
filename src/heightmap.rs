use crate::las_data::Limits;
use itertools::iproduct;
use log::info;
use remedian::RemedianBlock;
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

#[derive(Serialize, Deserialize)]
pub struct Heightmap<T: Clone + Copy> {
    pub data: Vec<T>,
    pub width: usize,
    pub height: usize,
    pub scale_z: f32,
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

impl<T: Copy + Clone> Heightmap<Option<T>> {
    pub fn proportion_of_empty_cells(&self) -> f64 {
        self.data.iter().filter(|x| x.is_none()).count() as f64 / self.data.len() as f64
    }
}

impl Heightmap<Option<f32>> {
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
            scale_z: self.scale_z,
        }
    }

    pub fn min_z(&self) -> f32 {
        self.data
            .iter()
            .filter_map(|x| *x)
            .min_by(|a, b| a.total_cmp(b))
            .unwrap()
    }

    pub fn max_z(&self) -> f32 {
        self.data
            .iter()
            .filter_map(|x| *x)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    }

    pub fn normalize_z_by(&self, min_z: f32, max_z: f32) -> Self {
        let adjust_z = -min_z;
        let delta = (max_z - min_z).abs();
        Heightmap {
            data: self
                .data
                .iter()
                .map(|x| match x {
                    Some(x) => {
                        assert!((x + adjust_z) / delta >= 0.);
                        Some((x + adjust_z) / delta)
                    }
                    None => None,
                })
                .collect(),
            width: self.width,
            height: self.height,
            scale_z: delta,
        }
    }

    pub fn of_nan_as_none(old: Heightmap<f32>) -> Self {
        Self {
            data: old
                .data
                .into_iter()
                .map(|x| if x.is_nan() { None } else { Some(x) })
                .collect(),
            width: old.width,
            height: old.height,
            scale_z: old.scale_z,
        }
    }

    pub fn into_nan_as_none(self: Heightmap<Option<f32>>) -> Heightmap<f32> {
        Heightmap {
            data: self
                .data
                .into_iter()
                .map(|x| match x {
                    Some(x) => x,
                    None => f32::NAN,
                })
                .collect(),
            width: self.width,
            height: self.height,
            scale_z: self.scale_z,
        }
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
            scale_z: 1.,
        }
    }

    pub fn add_base(&self, depth: f32) -> Self {
        Heightmap {
            data: self.data.iter().map(|x| x + depth).collect(),
            width: self.width,
            height: self.height,
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
            scale_z: max_z,
        }
    }

    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Self {
        Heightmap {
            data: self.data.iter().map(|x| f(*x)).collect(),
            width: self.width,
            height: self.height,
            scale_z: self.scale_z,
        }
    }
}

pub struct StreamingHeightmap {
    grid_zones: Vec<RemedianBlock<f32>>,
    grid_x: usize,
    grid_y: usize,
    ext_x: usize,
    ext_y: usize,
    limits: Limits,
}

impl StreamingHeightmap {
    pub fn new(limits: &Limits, pixels_per_distance_unit: f32) -> Self {
        let grid_x = ((limits.max_x - limits.min_x) * pixels_per_distance_unit).ceil() as usize;
        let grid_y = ((limits.max_y - limits.min_y) * pixels_per_distance_unit).ceil() as usize;
        info!("Derived GRID_X: {}, Derived GRID_Y: {}", grid_x, grid_y);

        let ext_x = grid_x + 1;
        let ext_y = grid_y + 1;
        let mut grid_zones = Vec::new();
        grid_zones.resize_with(ext_x * ext_y, || RemedianBlock::default());
        Self {
            grid_x,
            grid_y,
            ext_x,
            ext_y,
            limits: limits.clone(),
            grid_zones,
        }
    }

    pub fn add(&mut self, (px, py, pz): (f32, f32, f32)) {
        let x_ratio = (px - self.limits.min_x) / (self.limits.max_x - self.limits.min_x);
        let y_ratio = (py - self.limits.min_y) / (self.limits.max_y - self.limits.min_y);
        let grid_x = (x_ratio * self.grid_x as f32).floor() as usize;
        let grid_y = (y_ratio * self.grid_y as f32).floor() as usize;
        let zone = &mut self.grid_zones[(grid_y * self.ext_x) + grid_x];
        zone.add_sample_point(pz);
    }

    pub fn finalize(&self) -> Heightmap<Option<f32>> {
        info!("Constructed quantograms");

        let grid_zones: Vec<Option<f32>> = self
            .grid_zones
            .iter()
            .map(|grid_zone| grid_zone.median())
            .collect();

        info!("Summarized grid zones");

        Heightmap {
            data: grid_zones,
            width: self.ext_x,
            height: self.ext_y,
            scale_z: 1.,
        }
    }
}
