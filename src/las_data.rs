use las::{Read, Reader};
use log::info;
use std::ffi::OsStr;
use std::sync::mpsc::sync_channel;
use threadpool::ThreadPool;
use walkdir::WalkDir;

pub fn load_from_directory<F>(
    path: &str,
    (scale_x, scale_y, scale_z): (f32, f32, f32),
    max_threads: usize,
    mut f: F,
) where
    F: FnMut(f32, f32, f32) -> (),
{
    info!("Beginning iteration over all LAS data");

    let (sender, receiver) = sync_channel(1024 * 1024 * 1024);

    let pool = ThreadPool::new(max_threads);

    for entry in WalkDir::new(path)
        .max_depth(100)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path().extension() == Some(&OsStr::new("laz"))
                || e.path().extension() == Some(&OsStr::new("las"))
        })
    {
        let sender = sender.clone();
        pool.execute(move || {
            info!("Loading path: {:?}", entry.path());
            let mut reader = Reader::from_path(entry.path()).expect("Unable to open reader");

            for wrapped_point in reader.points() {
                let wrapped_point = wrapped_point.unwrap();

                let (x, y, z) = (
                    wrapped_point.x as f32 * scale_x,
                    wrapped_point.y as f32 * scale_y,
                    wrapped_point.z as f32 * scale_z,
                );

                sender.send((x, y, z)).unwrap();
            }

            drop(sender);
            info!("Finished");
        });
    }

    drop(sender);

    for (x, y, z) in receiver {
        f(x, y, z);
    }
}

#[derive(Debug, Default, Clone)]
pub struct Limits {
    pub min_x: f32,
    pub min_y: f32,
    pub min_z: f32,
    pub max_x: f32,
    pub max_y: f32,
    pub max_z: f32,
}

impl Limits {
    pub fn load_from_directory(path: &str, scalers: (f32, f32, f32), max_threads: usize) -> Self {
        let mut max_x: Option<f32> = None;
        let mut min_x: Option<f32> = None;
        let mut max_z: Option<f32> = None;
        let mut min_z: Option<f32> = None;
        let mut max_y: Option<f32> = None;
        let mut min_y: Option<f32> = None;
        println!("{}", path);
        load_from_directory(path, scalers, max_threads, |x, y, z| {
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
