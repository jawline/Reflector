use clap::Parser;
use log::info;
use rust_las_printer::heightmap::{Heightmap, InterpolationMode, StreamingHeightmap};
use rust_las_printer::las_data::{load_from_directory, LasType, Limits};
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

    #[arg(long)]
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

fn construct_heightmap(limits: &Limits, args: &Args, las_type: &LasType) -> Heightmap<Option<f64>> {
    let mut streamed = StreamingHeightmap::new(&limits, args.pixels_per_unit_dim);

    load_from_directory(
        &args.las_folder_path,
        (args.scale_x, args.scale_y, args.scale_z),
        args.max_threads,
        las_type,
        |x, y, z| {
            streamed.add((x, y, z));
        },
    );

    streamed.finalize().flip_y()
}

fn build_terrain_map(limits: &Limits, args: &Args) -> Heightmap<Option<f64>> {
    let mut grid_zones = construct_heightmap(&limits, &args, &LasType::Ground);

    grid_zones.building_ray_filler(5);
    grid_zones.flood_fill();

    info!("Doing hole filling");

    let grid_zones = (0..args.rounds_of_interpolated_hole_filling).fold(grid_zones, |acc, i| {
        info!("Neighbor filling round {}", i);
        acc.interpolate_missing_using_neighbors(
            InterpolationMode::Mean, // Percentile(0.1),
            args.consider_nearest_n_neighbors_for_interpolation,
        )
    });

    grid_zones
}

fn build_building_map(limits: &Limits, args: &Args) -> Heightmap<Option<f64>> {
    let mut grid_zones = construct_heightmap(&limits, &args, &LasType::Buildings);
    grid_zones.building_ray_filler(5);
    grid_zones
}

fn merge(terrain: &mut Heightmap<Option<f64>>, building: &Heightmap<Option<f64>>) {
    for x in 0..terrain.width {
        for y in 0..terrain.height {
            match building[(x, y)] {
                Some(v) => terrain[(x, y)] = Some(v),
                None => {}
            }
        }
    }
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

    let mut grid_zones = build_terrain_map(&limits, &args);
    let buildings = build_building_map(&limits, &args);
    merge(&mut grid_zones, &buildings);

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
