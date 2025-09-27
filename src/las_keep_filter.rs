use las::point::Classification;
use las::{Header, Read, Reader};
use log::info;
use std::collections::HashSet;
use std::ffi::OsStr;
use std::sync::mpsc::sync_channel;
use threadpool::ThreadPool;
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct LasKeepFilter {
    mapping: HashSet<u8>,
}

impl LasKeepFilter {
    pub fn new(entries: &[Classification]) -> Self {
        Self {
            mapping: HashSet::from_iter(entries.into_iter().map(|x| (*x).into())),
        }
    }

    pub fn default(include_unclassified: bool) -> Self {
        use Classification::*;
        if include_unclassified {
            Self::new(&[
                Ground,
                Building,
                RoadSurface,
                BridgeDeck,
                Rail,
                Water,
                Unclassified,
            ])
        } else {
            Self::new(&[Ground, Building, RoadSurface, BridgeDeck, Rail, Water])
        }
    }

    pub fn all() -> Self {
        Self {
            mapping: HashSet::from_iter(0..=255),
        }
    }

    pub fn contains(&self, classification: &Classification) -> bool {
        self.mapping.contains(&(*classification).into())
    }
}
