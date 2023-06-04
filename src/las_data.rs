use las::{Read, Reader};
use log::{debug, info};
use std::ffi::OsStr;
use walkdir::WalkDir;

pub struct LasData {
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
    pub points: Vec<(f64, f64, f64)>,
}

impl LasData {
    pub fn load_from_directory(path: &str) -> Self {
        let mut max_x = None;
        let mut min_x = None;
        let mut max_z = None;
        let mut min_z = None;
        let mut max_y = None;
        let mut min_y = None;

        println!("Beginning first pass");

        let mut points = Vec::new();

        for entry in WalkDir::new(path.clone())
            .max_depth(100)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .filter(|e| e.path().extension() == Some(&OsStr::new("las")))
        {
            debug!("Loading path: {:?}", entry.path());
            let mut reader = Reader::from_path(entry.path()).expect("Unable to open reader");

            for wrapped_point in reader.points() {
                let wrapped_point = wrapped_point.unwrap();

                max_x = Some(max_x.unwrap_or(wrapped_point.x).max(wrapped_point.x));
                max_y = Some(max_y.unwrap_or(wrapped_point.y).max(wrapped_point.y));
                max_z = Some(max_z.unwrap_or(wrapped_point.z).max(wrapped_point.z));
                min_x = Some(min_x.unwrap_or(wrapped_point.x).min(wrapped_point.x));
                min_y = Some(min_y.unwrap_or(wrapped_point.y).min(wrapped_point.y));
                min_z = Some(min_z.unwrap_or(wrapped_point.z).min(wrapped_point.z));

                points.push((wrapped_point.x, wrapped_point.y, wrapped_point.z));
            }
        }

        let (min_x, max_x, min_y, max_y, min_z, max_z) = (
            min_x.unwrap(),
            max_x.unwrap(),
            min_y.unwrap(),
            max_y.unwrap(),
            min_z.unwrap(),
            max_z.unwrap(),
        );

        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            min_z,
            max_z,
            points,
        }
    }
}
