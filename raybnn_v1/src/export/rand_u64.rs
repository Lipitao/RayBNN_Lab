extern crate arrayfire;

use crate::export::rand_f64::single_random_uniform;

pub fn random_uniform_range(max_size: u64) -> u64 {
    let rand_number = single_random_uniform();

    let min_idx = (rand_number * (max_size as f64)) as u64;

    return min_idx;
}
