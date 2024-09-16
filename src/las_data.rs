use las::{Read, Reader};
use log::info;
use std::ffi::OsStr;
use walkdir::WalkDir;

pub fn load_from_directory<F>(path: &str, (scale_x, scale_y, scale_z): (f64, f64, f64), mut f: F)
where
    F: FnMut(f64, f64, f64) -> (),
{
    info!("Beginning iteration over all LAS data");

    for entry in WalkDir::new(path)
        .max_depth(100)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| e.path().extension() == Some(&OsStr::new("las")))
    {
        info!("Loading path: {:?}", entry.path());
        let mut reader = Reader::from_path(entry.path()).expect("Unable to open reader");

        for wrapped_point in reader.points() {
            let wrapped_point = wrapped_point.unwrap();

            let (x, y, z) = (
                wrapped_point.x * scale_x,
                wrapped_point.y * scale_y,
                wrapped_point.z * scale_z,
            );

            f(x, y, z);
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Limits {
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
}

impl Limits {
    pub fn load_from_directory(path: &str, scalers: (f64, f64, f64)) -> Self {
        let mut max_x: Option<f64> = None;
        let mut min_x: Option<f64> = None;
        let mut max_z: Option<f64> = None;
        let mut min_z: Option<f64> = None;
        let mut max_y: Option<f64> = None;
        let mut min_y: Option<f64> = None;
        load_from_directory(path, scalers, |x, y, z| {
            max_x = Some(max_x.unwrap_or(x).max(x));
            max_y = Some(max_y.unwrap_or(y).max(y));
            max_z = Some(max_z.unwrap_or(z).max(z));
            min_x = Some(min_x.unwrap_or(x).min(x));
            min_y = Some(min_y.unwrap_or(y).min(y));
            min_z = Some(min_z.unwrap_or(z).min(z));
        });

        Self {
            min_x: min_x.unwrap(),
            max_x: max_x.unwrap(),
            min_y: min_y.unwrap(),
            max_y: max_y.unwrap(),
            min_z: min_z.unwrap(),
            max_z: max_z.unwrap(),
        }
    }
}
