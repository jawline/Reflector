use las::point::Classification;
use las::{Header, Read, Reader};
use log::info;
use std::ffi::OsStr;
use std::sync::mpsc::sync_channel;
use threadpool::ThreadPool;
use walkdir::WalkDir;

// We re-expose classification because Las_data does not make it sortable or hashable
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum LasType {
    Ground,
    GroundAndWater,
    Buildings,
    Water,
    GroundAndBuildingsAndWater,
    All,
}

impl LasType {
    fn contains(&self, classification: &Classification) -> bool {
        use Classification::*;
        match self {
            LasType::Ground => match classification {
                // TODO: Make this customizable - I've found anecodtally that depending on dataset
                // ground is often classified as low / medium vegetation
                Ground => true,
                _ => false,
            },
            LasType::Buildings => match classification {
                Building | RoadSurface | BridgeDeck | Rail | Unclassified=> true,
                // TODO: Consider these, but I think they aren't good things to add | WireGuard | WireConductor | TransmissionTower | WireStructureConnector |
                _ => false,
            },
            LasType::Water => match classification {
                Water => true,
                _ => false,
            },
            LasType::GroundAndWater => {
                LasType::Ground.contains(classification) || LasType::Water.contains(classification)
            }
            LasType::GroundAndBuildingsAndWater => {
                LasType::Ground.contains(classification)
                    || LasType::Buildings.contains(classification)
                    || LasType::Water.contains(classification)
            }
            LasType::All => true,
        }
    }
}

pub fn find_a_header(path: &str) -> Header {
    let mut header = None;
    for entry in WalkDir::new(path)
        .max_depth(100)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path().extension() == Some(&OsStr::new("las"))
                || e.path().extension() == Some(&OsStr::new("laz"))
        })
    {
        header = Some(
            Reader::from_path(entry.path())
                .expect("Unable to open reader")
                .header()
                .clone(),
        );
        // TODO: Scan all headers and check they are the same.
        break;
    }

    header.expect("No files found")
}

pub fn load_from_directory_points<F>(path: &str, max_threads: usize, mut f: F)
where
    F: FnMut(las::point::Point) -> (),
{
    info!("Beginning iteration over all LAS data");

    let (sender, receiver) = sync_channel(1024 * 1024);

    let pool = ThreadPool::new(max_threads);

    for entry in WalkDir::new(path)
        .max_depth(100)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path().extension() == Some(&OsStr::new("las"))
                || e.path().extension() == Some(&OsStr::new("laz"))
        })
    {
        let sender = sender.clone();
        pool.execute(move || {
            info!("Loading path: {:?}", entry.path());
            let mut reader = Reader::from_path(entry.path()).expect("Unable to open reader");

            for wrapped_point in reader.points() {
                sender.send(wrapped_point.expect("File error")).unwrap();
            }

            drop(sender);
            info!("Finished");
        });
    }

    drop(sender);

    for pt in receiver {
        f(pt);
    }
}

// TODO: Refactor in terms of load_from_directory_points
pub fn load_from_directory<F>(path: &str, max_threads: usize, las_type: &LasType, mut f: F)
where
    F: FnMut(f32, f32, f32) -> (),
{
    info!("Beginning iteration over all LAS data");

    let (sender, receiver) = sync_channel(1024 * 1024);

    let pool = ThreadPool::new(max_threads);

    for entry in WalkDir::new(path)
        .max_depth(100)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path().extension() == Some(&OsStr::new("las"))
                || e.path().extension() == Some(&OsStr::new("laz"))
        })
    {
        let sender = sender.clone();
        let las_type = las_type.clone();
        pool.execute(move || {
            info!("Loading path: {:?}", entry.path());
            let mut reader = Reader::from_path(entry.path()).expect("Unable to open reader");

            for wrapped_point in reader.points() {
                let wrapped_point = wrapped_point.unwrap();

                let (x, y, z) = (
                    wrapped_point.x as f32,
                    wrapped_point.y as f32,
                    wrapped_point.z as f32,
                );

                let is_correct_classification = las_type.contains(&wrapped_point.classification);
                if is_correct_classification {
                    sender.send((x, y, z)).unwrap();
                }
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
    pub fn load_from_directory(path: &str, max_threads: usize) -> Self {
        let mut max_x: Option<_> = None;
        let mut min_x: Option<_> = None;
        let mut max_z: Option<_> = None;
        let mut min_z: Option<_> = None;
        let mut max_y: Option<_> = None;
        let mut min_y: Option<_> = None;
        load_from_directory(path, max_threads, &LasType::All, |x, y, z| {
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
