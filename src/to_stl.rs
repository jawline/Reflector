use crate::renderer::types::Model;
use bevy::math::Vec3;
use log::debug;
use stl_io::{Triangle, Vector};

/// Converts a model to stl::io Triangle's for writing to stl.
pub fn to_stl(model: &Model) -> Vec<Triangle> {
    debug!("Creating stl from &Model");
    let mut output = Vec::new();

    for triangle in &model.triangles {
        let vertex1 = model.vertices[triangle[0]];
        let vertex2 = model.vertices[triangle[1]];
        let vertex3 = model.vertices[triangle[2]];

        // CCW order
        let vertex1 = Vec3::from(vertex1);
        let vertex2 = Vec3::from(vertex2);
        let vertex3 = Vec3::from(vertex3);

        let u = vertex2 - vertex1;
        let v = vertex3 - vertex1;
        let n = u.cross(v);
        let normal = n.normalize();

        //println!("{:?} {:?} {:?}", vertex1, vertex2, vertex3);
        //println!("U {:?} V {:?}", u, v);
        //println!("N {:?}", n);
        //println!("{:?}", normal);

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
