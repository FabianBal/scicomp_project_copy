
pub struct Dense {
    pub data: Vec<f64>,
    pub shape: (usize, usize)
}


impl Dense {
    pub fn new_zeros(shape: (usize, usize)) -> Self {
        Dense{data: vec![0.;shape.0*shape.1], shape}
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[self.shape.1 *i + j]
    }

    pub fn set(&mut self, i: usize, j: usize, x: f64) {
        self.data[self.shape.1 *i + j] = x;
    }

    pub fn print(&self) {
        println!("Dense ({},{})-matrix", self.shape.0, self.shape.1);
        for i in 0..self.shape.0 {
            // Ja geht auch effizienter, indem man direkt die row plottet
            // weiß aber grad nicht, ob man slices auch außerhalb des
            // Debug-Mode easy printen kann
            for j in 0..self.shape.1 {
                print!("{}\t", self.get(i, j));
            }
            println!();
        }
    }
}