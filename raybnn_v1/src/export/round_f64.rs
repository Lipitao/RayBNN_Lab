extern crate arrayfire;

use rayon::prelude::*;

const TEN: f64 = 10.0;

pub fn rscalar(input: f64, decimal: u64) -> f64 {
    let places = TEN.powf(decimal as f64);
    (input * places).round() / places
}

pub fn rvector(input: &Vec<f64>, decimal: u64) -> Vec<f64> {
    input
        .par_iter()
        .map(|&x| rscalar(x, decimal))
        .collect::<Vec<f64>>()
}
