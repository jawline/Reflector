use serde_repr::{Deserialize_repr, Serialize_repr};

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum ClassificationType {
    Unknown = 0,
    GroundLayer = 1,
    BuildingsOrVegetationLayer = 2,
}
