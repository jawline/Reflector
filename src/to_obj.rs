//use crate::to_3d_model::Model;
//use crate::to_3d_model::TriangleCCW;
//use bevy::math::Vec3;
use crate::renderer::types::Model;
use log::debug;
use threecrate_core::mesh::TriangleMesh;
use threecrate_core::Point3;

/// Converts a model to stl::io Triangle's for writing to stl.
pub fn to_obj(model: Model) -> TriangleMesh {
    debug!("Creating obj from Model");

    TriangleMesh {
        vertices: model
            .vertices
            .into_iter()
            .map(|[x, y, z]| Point3::new(x, y, z))
            .collect(),
        faces: model.triangles.into_iter().collect(),
        colors: None,
        normals: None,
    }
}
