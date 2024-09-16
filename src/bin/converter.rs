use clap::Parser;
use log::info;
use rust_las_printer::heightmap::{Heightmap, StreamingHeightmap};
use rust_las_printer::las_data::{load_from_directory, Limits};
use rust_las_printer::to_3d_model::Model;
use rust_las_printer::to_stl::to_stl;
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

    #[arg(long, default_value_t = false)]
    write_to_bin: bool,

    #[arg(long, default_value_t = false)]
    write_to_stl: bool,

    #[arg(short, long, default_value_t = 0.0)]
    base_depth: f64,

    #[arg(short, long)]
    override_base_depth_for_tiles_with_no_data: Option<f64>,

    #[arg(long, default_value_t = 1.0)]
    scale_x: f64,
    #[arg(long, default_value_t = 1.0)]
    scale_y: f64,
    #[arg(long, default_value_t = 1.0)]
    scale_z: f64,

    #[arg(long, default_value_t = 16)]
    max_threads: usize,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    println!("First pass, collecting limits");
    let limits = Limits::load_from_directory(
        &args.las_folder_path,
        (args.scale_x, args.scale_y, args.scale_z),
        args.max_threads,
    );

    info!(
        "Bounds: {} {} {} {} {} {}",
        limits.min_x, limits.max_x, limits.min_y, limits.max_y, limits.min_z, limits.max_z
    );

    println!("Main pass, summarizing grid squares");

    let mut streamed = StreamingHeightmap::new(&limits, args.pixels_per_unit_dim);

    load_from_directory(
        &args.las_folder_path,
        (args.scale_x, args.scale_y, args.scale_z),
        args.max_threads,
        |x, y, z| {
            streamed.add((x, y, z));
        },
    );

    let grid_zones = streamed.finalize();

    info!("Flipping the Y axis");
    let grid_zones = grid_zones.flip_y();

    info!("Doing hole filling");

    let grid_zones = (0..args.rounds_of_interpolated_hole_filling).fold(grid_zones, |acc, i| {
        info!("Neighbor filling round {}", i);
        acc.interpolate_missing_using_neighbors(args.consider_nearest_n_neighbors_for_interpolation)
    });

    // Here every point will be some
    info!("Normalizing Z axis");
    let grid_zones = grid_zones.fill_none_with_zero_and_add_base(
        args.base_depth,
        args.override_base_depth_for_tiles_with_no_data
            .unwrap_or(args.base_depth),
    );

    if args.write_to_bin && args.write_to_stl {
        panic!("We expect only one of write_to_bin or write_to_stl to be set");
    }

    if args.write_to_stl {
        let model = Model::of_heightmap(&grid_zones);
        let mesh = to_stl(&model);
        let mut file = File::create(args.output_path).unwrap();
        stl_io::write_stl(&mut file, mesh.into_iter()).unwrap();
    } else if args.write_to_bin {
        let file = File::create(args.output_path).unwrap();
        let mut writer = BufWriter::new(file);
        writer
            .write(&postcard::to_stdvec::<Heightmap<f64>>(&grid_zones).unwrap())
            .unwrap();
    } else {
        let max_z = grid_zones.max_z();
        let grid_zones = grid_zones.normalize_z_by(max_z).to_u8(args.max_y_is_low);
        grid_zones.write_to_png(&args.output_path);
    }
}
