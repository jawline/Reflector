use clap::Parser;
use log::info;
use rust_las_printer::las_data::{find_a_header, load_from_directory_points, Limits};
use std::cmp::min;

use las::{Write, Writer};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    input_directory: String,

    #[arg(short, long)]
    output_directory: String,

    #[arg(short, long)]
    files_x_dim: usize,

    #[arg(short, long)]
    files_y_dim: usize,
}

fn main() {
    env_logger::init();

    let args = Args::parse();
    let header = find_a_header(&args.input_directory);

    println!("First pass, collecting limits");
    let limits = Limits::load_from_directory(&args.input_directory, (1., 1., 1.), 1);

    info!(
        "Bounds: {} {} {} {} {} {}",
        limits.min_x, limits.max_x, limits.min_y, limits.max_y, limits.min_z, limits.max_z
    );

    println!("Main pass, summarizing grid squares");

    let x_div = (limits.max_x - limits.min_x) / (args.files_x_dim as f32);
    let y_div = (limits.max_y - limits.min_y) / (args.files_y_dim as f32);

    println!("XDIV: {} YDIV: {}", x_div, y_div);

    let mut file_grid = Vec::new();

    for x in 0..(args.files_x_dim as usize) {
        file_grid.push(Vec::new());
        for y in 0..(args.files_y_dim as usize) {
            // LAZ for compression
            let file_name = format!("{}/x_{}_y_{}.laz", args.output_directory, x, y);
            let writer = Writer::from_path(file_name, header.clone()).unwrap();
            file_grid[x].push(writer)
        }
    }

    load_from_directory_points(&args.input_directory, 1, |point| {
        // Relative to min pt
        let x_pt = point.x as f32 - limits.min_x;
        let y_pt = point.y as f32 - limits.min_y;

        // Chunk index
        let x_div = min((x_pt / x_div) as usize, args.files_x_dim - 1);
        let y_div = min((y_pt / y_div) as usize, args.files_y_dim - 1);

        let writer = &mut file_grid[x_div as usize][y_div as usize];
        writer.write(point).unwrap();
    });
}
