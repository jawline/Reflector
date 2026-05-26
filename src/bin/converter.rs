use clap::Parser;
use log::{error, info, warn};
use serde_pickle::{DeOptions, SerOptions};
use std::fs::{read, File};
use threecrate_io::{obj::ObjWriter, MeshWriter};

use rust_las_printer::{
    las_converter::{
        classification::ClassificationType,
        heightmap::{Heightmap, HeightmapAndClassification, InterpolationMode, StreamingHeightmap},
        las_data::{load_from_directory, Limits},
        las_keep_filter::LasKeepFilter,
    },
    renderer::to_3d_model,
    to_obj::to_obj,
    to_stl::to_stl,
};

#[derive(PartialEq, Parser, Eq, Debug, Clone)]
enum Format {
    Stl,
    Obj,
}

#[derive(Parser, Debug)]
struct GenerateHeightmap {
    #[arg(short, long)]
    las_folder_path: String,

    #[arg(short, long)]
    output_path: String,

    #[arg(short, long, default_value_t = 0.25)]
    pixels_per_unit_dim: f32,

    #[arg(short, long, default_value_t = false)]
    max_y_is_low: bool,

    #[arg(long, default_value_t = 16)]
    max_threads: usize,

    #[arg(long, default_value_t = false)]
    assume_unclassified_are_buildings_or_vegetation: bool,

    #[arg(long, default_value_t = false)]
    include_vegetation: bool,
}

#[derive(Parser, Debug)]
struct Render {
    #[arg(short, long)]
    read_from: String,

    #[arg(short, long)]
    write_to: String,

    #[clap(subcommand)]
    format: Format,

    mode: to_3d_model::Mode,

    #[arg(short, long, default_value_t = 0.0)]
    base_depth: f32,

    #[arg(short, long, default_value_t = 1)]
    rounds_of_interpolated_hole_filling: usize,

    #[arg(short, long, default_value_t = 16)]
    consider_nearest_n_neighbors_for_interpolation: usize,

    #[arg(long)]
    override_base_depth_for_tiles_with_no_data: Option<f32>,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
enum Args {
    GenerateHeightmap(GenerateHeightmap),
    Render(Render),
}

fn construct_heightmap(
    limits: &Limits,
    args: &GenerateHeightmap,
    filter: &LasKeepFilter,
) -> Heightmap<Option<f32>> {
    let mut streamed = StreamingHeightmap::new(&limits, args.pixels_per_unit_dim);

    load_from_directory(
        &args.las_folder_path,
        args.max_threads,
        filter,
        |x, y, z| {
            streamed.add((x, y, z));
        },
    );

    streamed.finalize()
}

fn build_terrain_map(limits: &Limits, args: &GenerateHeightmap) -> Heightmap<Option<f32>> {
    let grid_zones = construct_heightmap(&limits, &args, &LasKeepFilter::ground_layer());
    grid_zones
}

fn build_building_map(limits: &Limits, args: &GenerateHeightmap) -> Heightmap<Option<f32>> {
    let filter = LasKeepFilter::building_layer();

    let filter = if args.assume_unclassified_are_buildings_or_vegetation {
        filter.add_unclassified()
    } else {
        filter
    };

    let filter = if args.include_vegetation {
        filter.add_vegetation()
    } else {
        filter
    };

    let grid_zones = construct_heightmap(&limits, &args, &filter);
    grid_zones
}

fn merge(terrain: &mut Heightmap<Option<f32>>, building: &Heightmap<Option<f32>>) {
    for x in 0..terrain.width {
        for y in 0..terrain.height {
            match building[(x, y)] {
                Some(v) => terrain[(x, y)] = Some(v),
                None => {}
            }
        }
    }
}

fn build_classification_layer(
    terrain: &Heightmap<Option<f32>>,
    building: &Heightmap<Option<f32>>,
) -> Heightmap<ClassificationType> {
    Heightmap {
        data: (0..terrain.data.len())
            .map(|i| match (terrain.data[i], building.data[i]) {
                (_, Some(_)) => ClassificationType::BuildingsOrVegetationLayer,
                (Some(_), _) => ClassificationType::GroundLayer,
                (None, None) => ClassificationType::Unknown,
            })
            .collect(),
        width: terrain.width,
        height: terrain.height,
        scale_z: 1.,
    }
}

fn generate_heightmap(args: GenerateHeightmap) {
    info!("First pass, collecting limits");
    let limits = Limits::load_from_directory(&args.las_folder_path, args.max_threads);

    info!(
        "Bounds: {} {} {} {} {} {}",
        limits.min_x, limits.max_x, limits.min_y, limits.max_y, limits.min_z, limits.max_z
    );

    info!("Main pass, summarizing grid squares");

    // We merge the medians for the buildings and the medians for terrain and then make sure
    // the buildings take precedent. I have found this to be anecdotally better than just a
    // median since you get less noisy data but don't end up with weird looking underpasses.
    let mut grid_zones = build_terrain_map(&limits, &args);
    let buildings = build_building_map(&limits, &args);

    let classification = build_classification_layer(&grid_zones, &buildings);
    merge(&mut grid_zones, &buildings);

    let proportion_of_empty_cells = grid_zones.proportion_of_empty_cells();
    info!("Proportion of empty cells: {}", proportion_of_empty_cells);

    if proportion_of_empty_cells > 0.5 {
        warn!("BAD INPUT: With this upscaling, more than 50% of the pixels are none.");
    }

    if grid_zones.proportion_of_empty_cells() > 0.75 {
        error!("REJECTING INPUT DUE TO HIGH PROPORTION OF ERRORS");
    } else {
        let mut file = File::create(args.output_path).unwrap();
        let min_z = grid_zones.min_z();
        let max_z = grid_zones.max_z();

        info!("Normalizing min={} max={}", min_z, max_z);
        let grid_zones = grid_zones.normalize_z_by(min_z, max_z);
        serde_pickle::to_writer(
            &mut file,
            &HeightmapAndClassification {
                heightmap: grid_zones,
                classification,
            },
            SerOptions::new(),
        )
        .unwrap();
    }
}

fn render(args: Render) {
    let grid_zones = read(&args.read_from).unwrap();
    let grid_zones: Heightmap<Option<f32>> =
        serde_pickle::from_slice(&grid_zones, DeOptions::new()).unwrap();

    info!("Doing hole filling");

    let grid_zones = (0..args.rounds_of_interpolated_hole_filling).fold(grid_zones, |acc, i| {
        info!("Neighbor filling round {}", i);
        acc.interpolate_missing_using_neighbors(
            InterpolationMode::Min,
            args.consider_nearest_n_neighbors_for_interpolation,
        )
    });

    // Here every point will be some
    info!("Normalizing Z axis");
    let grid_zones = grid_zones.fill_none_with_zero_and_add_base(
        args.base_depth,
        args.override_base_depth_for_tiles_with_no_data
            .unwrap_or(args.base_depth),
    );

    let model = to_3d_model::of_heightmap(&grid_zones, &args.mode);

    match args.format {
        Format::Stl => {
            let mesh = to_stl(&model);
            let mut file = File::create(args.write_to).unwrap();
            stl_io::write_stl(&mut file, mesh.into_iter()).unwrap()
        }
        Format::Obj => {
            let mesh = to_obj(model);
            ObjWriter::write_mesh(&mesh, args.write_to).unwrap();
        }
    };
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    match args {
        Args::GenerateHeightmap(args) => {
            generate_heightmap(args);
        }
        Args::Render(args) => {
            render(args);
        }
    }
}
