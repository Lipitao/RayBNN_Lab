extern crate arrayfire;

use rayon::prelude::*;

const TEN: f32 = 10.0;

pub fn rscalar(input: f32, decimal: u64) -> f32 {
    let places = TEN.powf(decimal as f32);
    (input * places).round() / places
}

pub fn rvector(input: &Vec<f32>, decimal: u64) -> Vec<f32> {
    input
        .par_iter()
        .map(|&x| rscalar(x, decimal))
        .collect::<Vec<f32>>()
}
