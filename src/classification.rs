#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum ClassificationType {
    Unknown,
    GroundLayer,
    BuildingsLayer,
    Meta,
}

pub mod las {
    use super::ClassificationType;
    use las::point::Classification;

    pub struct LasToClassificationFilter {
        unclassified_as: ClassificationType,
    }

    impl LasToClassificationFilter {
        pub fn classify(&self, classification: &Classification) -> ClassificationType {
            use Classification::*;
            use ClassificationType::*;

            // TODO: Consider training with a RoadLayer, but this might not be great as a lot of data
            // does not label buildings and roads distinctly.
            match classification {
                Ground | LowVegetation | MediumVegetation | HighVegetation | Water => GroundLayer,
                Building
                | RoadSurface
                | BridgeDeck
                | Rail
                | WireGuard
                | WireConductor
                | TransmissionTower
                | WireStructureConnector => BuildingsLayer,
                Unclassified | CreatedNeverClassified => self.unclassified_as,
                LowPoint | ModelKeyPoint | HighNoise | Reserved(_) | UserDefinable(_) => Meta,
            }
        }
    }
}
