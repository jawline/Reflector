use super::bag_of_cubes;
use super::marching_cubes;
use super::terrain;
use super::types::Model;
use crate::heightmap::Heightmap;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub enum Mode {
    Terrain,
    BagOfCubes,
    MarchingCubes { z_steps: usize },
}

impl Mode {
    pub fn default() -> Self {
        Self::MarchingCubes { z_steps: 512 }
    }

    pub fn mode_names() -> HashMap<String, Mode> {
        HashMap::from([
            ("terrain".to_string(), Mode::Terrain),
            ("bag-of-cubes".to_string(), Mode::BagOfCubes),
            (
                "marching-cubes-low".to_string(),
                Mode::MarchingCubes { z_steps: 128 },
            ),
            (
                "marching-cubes".to_string(),
                Mode::MarchingCubes { z_steps: 512 },
            ),
        ])
    }

    pub fn of_string(s: &str) -> Mode {
        *Self::mode_names().get(s).unwrap()
    }
}

pub fn of_heightmap(heightmap: &Heightmap<f32>, mode: &Mode) -> Model {
    match mode {
        Mode::BagOfCubes => bag_of_cubes::of_heightmap(heightmap),
        Mode::MarchingCubes { z_steps } => marching_cubes::of_heightmap(heightmap, *z_steps),
        Mode::Terrain => terrain::of_heightmap(heightmap),
    }
}
