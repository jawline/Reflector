use super::bag_of_cubes;
use super::marching_cubes;
use super::terrain;
use super::types::Model;
use crate::heightmap::Heightmap;
use clap::{Parser, ValueEnum};

#[derive(PartialEq, Parser, Eq, Debug, Clone, Copy, ValueEnum, Default)]
#[clap(rename_all = "kebab-case")]
pub enum Mode {
    Terrain,
    BagOfCubes,
    MarchingCubesLow,
    #[default]
    MarchingCubesMedium,
    MarchingCubesHigh,
}

pub fn of_heightmap(heightmap: &Heightmap<f32>, mode: &Mode) -> Model {
    match mode {
        Mode::BagOfCubes => bag_of_cubes::of_heightmap(heightmap),
        Mode::MarchingCubesLow => marching_cubes::of_heightmap(heightmap, 128),
        Mode::MarchingCubesMedium => marching_cubes::of_heightmap(heightmap, 256),
        Mode::MarchingCubesHigh => marching_cubes::of_heightmap(heightmap, 512),
        Mode::Terrain => terrain::of_heightmap(heightmap),
    }
}
