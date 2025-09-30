use las::point::Classification;
use std::collections::HashSet;
use Classification::*;

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

    pub fn add_unclassified(self) -> Self {
        let mut mapping = self.mapping;
        mapping.insert(Unclassified.into());
        mapping.insert(CreatedNeverClassified.into());
        Self { mapping }
    }

    pub fn ground_layer() -> Self {
        Self::new(&[Ground, Water])
    }

    pub fn building_layer() -> Self {
        Self::new(&[Building, RoadSurface, BridgeDeck, Rail])
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
