use crate::to_3d_model::Model;
use bevy::math::Vec3;
use log::debug;
use stl_io::{Triangle, Vector};

/// Converts a model to stl::io Triangle's for writing to stl.
pub fn to_stl(model: &Model) -> Vec<Triangle> {
    debug!("Creating stl from &Model");
    let mut output = Vec::new();

    for i in (0..model.indices.len()).step_by(3) {
        let index1 = model.indices[i] as usize;
        let index2 = model.indices[i + 1] as usize;
        let index3 = model.indices[i + 2] as usize;
        let normal: [f32; 3] = ((Vec3::from(model.normals[index1])
            + Vec3::from(model.normals[index2])
            + Vec3::from(model.normals[index3]))
            / 3.)
            .into();

        let triangle = Triangle {
            normal: Vector::new(normal),
            vertices: [
                Vector::new(model.vertices[index1].clone()),
                Vector::new(model.vertices[index2].clone()),
                Vector::new(model.vertices[index3].clone()),
            ],
        };

        output.push(triangle);
    }

    output
}
