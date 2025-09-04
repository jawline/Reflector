pub type Point = [f32; 3];

pub type Triangle = [usize; 3];

pub struct Model {
    pub vertices: Vec<Point>,
    pub triangles: Vec<Triangle>,
}

pub fn scale_point(p: &Point, q: &Point) -> Point {
    [p[0] * q[0], p[1] * q[1], p[2] * q[2]]
}

impl Model {
    pub fn scale(&mut self, (scale_x, scale_y, scale_z): (f32, f32, f32)) {
        self.vertices = self
            .vertices
            .iter()
            .map(|x| scale_point(x, &[scale_x, scale_y, scale_z]))
            .collect();
    }
}
