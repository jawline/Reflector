use crate::to_3d_model::Model;
use bevy::math::Vec3;
use log::debug;
use stl_io::{Triangle, Vector};

/// Converts a model to stl::io Triangle's for writing to stl.
pub fn to_stl(model: &Model) -> Vec<Triangle> {
    debug!("Creating stl from &Model");
    let mut output = Vec::new();

    for triangle in &model.triangles {
        // CCW order
        let vertex1 = Vec3::from(triangle[0].clone());
        let vertex2 = Vec3::from(triangle[1].clone());
        let vertex3 = Vec3::from(triangle[2].clone());

        let u = vertex2 - vertex1;
        let v = vertex3 - vertex1;
        let n = u.cross(v);
        let normal = n.normalize();

        println!("{:?} {:?} {:?}", vertex1, vertex2, vertex3);
        println!("U {:?} V {:?}", u, v);
        println!("N {:?}", n);
        println!("{:?}", normal);

        let triangle = Triangle {
            normal: Vector::new(normal.into()),
            vertices: [
                Vector::new(vertex1.into()),
                Vector::new(vertex2.into()),
                Vector::new(vertex3.into()),
            ],
        };

        output.push(triangle);
    }

    output
}
