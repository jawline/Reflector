use clap::Parser;
use log::info;
use rust_las_printer::heightmap::{las_data_to_opt_height_map, Heightmap};
use rust_las_printer::las_data::LasData;
use std::{
    fs::File,
    io::{BufWriter, Write},
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    las_folder_path: String,

    #[arg(short, long)]
    output_path: String,

    #[arg(short, long, default_value_t = 0.25)]
    pixels_per_unit_dim: f64,

    #[arg(short, long, default_value_t = 1)]
    rounds_of_interpolated_hole_filling: usize,

    #[arg(short, long, default_value_t = 16)]
    consider_nearest_n_neighbors_for_interpolation: usize,

    #[arg(short, long, default_value_t = false)]
    max_y_is_low: bool,

    #[arg(short, long, default_value_t = false)]
    write_to_bin: bool,

    #[arg(short, long, default_value_t = 0.0)]
    base_depth: f64,
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

    let grid_zones = las_data_to_opt_height_map(&data, args.pixels_per_unit_dim);

    info!("Flipping the Y axis");
    let grid_zones = grid_zones.flip_y();

    info!("Doing hole filling");

    let grid_zones = (0..args.rounds_of_interpolated_hole_filling).fold(grid_zones, |acc, i| {
        info!("Neighbor filling round {}", i);
        acc.interpolate_missing_using_neighbors(args.consider_nearest_n_neighbors_for_interpolation)
    });

    // Here every point will be some
    info!("Normalizing Z axis");
    let grid_zones = grid_zones.fill_none_with_zero();

    info!("Writing to file");

    let grid_zones = grid_zones.add_base(args.base_depth);

    if !args.write_to_bin {
        let max_z = grid_zones.max_z();
        let grid_zones = grid_zones.normalize_z_by(max_z).to_u8(args.max_y_is_low);
        grid_zones.write_to_png(&args.output_path);
    } else {
        let file = File::create(args.output_path).unwrap();
        let mut writer = BufWriter::new(file);
        writer
            .write(&postcard::to_stdvec::<Heightmap<f64>>(&grid_zones).unwrap())
            .unwrap();
    }
}
