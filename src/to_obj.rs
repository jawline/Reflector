use crate::to_3d_model::Model;
//use crate::to_3d_model::TriangleCCW;
//use bevy::math::Vec3;
use log::debug;
//use stl_io::{Triangle, Vector};
use threecrate_core::mesh::TriangleMesh;
//use threecrate_core::Point3;

/// Converts a model to stl::io Triangle's for writing to stl.
pub fn to_obj(_model: &Model) -> TriangleMesh {
    debug!("Creating obj from Model");
    unimplemented!();

    //TriangleMesh {
    //    vertices: model
    //        .vertices
    //        .iter()
    //        .map(|TriangleCCW { d: [x, y, z] }| Point3::new(*x, *y, *z))
    //        .collect(),
    //    faces: unimplemented!(),
    //    colors: None,
    //    normals: None,
    //}
}
