mod heightmap;
mod las_data;

use crate::heightmap::las_data_to_opt_height_map;
use crate::las_data::LasData;
use clap::Parser;
use log::info;

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

    #[arg(short, long, default_value_t = 16)]
    consider_nearest_n_neighbors_for_interpolation: usize,
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
        grid_zones = grid_zones.interpolate_missing_using_neighbors(
            args.consider_nearest_n_neighbors_for_interpolation,
        );
    }

    // Here every point will be some
    info!("Normalizing Z axis");
    let grid_zones = grid_zones.normalize_z_by_and_fill_none_with_zero(data.max_z);

    info!("Writing to file");
    grid_zones.write_to_png(&args.output_path);
}
