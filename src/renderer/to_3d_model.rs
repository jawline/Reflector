use super::bag_of_cubes;
use super::marching_cubes;
use super::types::Model;
use crate::heightmap::Heightmap;

pub enum Mode {
    Terrain,
    BagOfCubes,
    MarchingCubes { z_steps: usize },
}

impl Mode {
    pub fn default() -> Self {
        Self::MarchingCubes { z_steps: 512 }
    }
}

pub fn of_heightmap(heightmap: &Heightmap<f32>, mode: &Mode) -> Model {
    match mode {
        Mode::BagOfCubes => bag_of_cubes::of_heightmap(heightmap),
        Mode::MarchingCubes { z_steps } => marching_cubes::of_heightmap(heightmap, *z_steps),
        Mode::Terrain => todo!()
    }
}
