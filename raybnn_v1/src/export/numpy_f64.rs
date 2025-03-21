extern crate arrayfire;
use crate::neural::network_f64::network_metadata_type;
use nohash_hasher;
use std::collections::HashMap;

use rayon::array;
use rayon::prelude::*;

use std::fs::File;
use std::io::Write;

use std::fs;

use crate::export::dataloader_i32::str_to_vec as str_to_vec_i32;
use crate::export::dataloader_i32::vec_to_str as vec_to_str_i32;

use crate::export::dataloader_u64::str_to_vec_cpu as str_to_vec_cpu_u64;
use crate::export::dataloader_u64::vec_cpu_to_str as vec_cpu_to_str_u64;

use crate::neural::network_f64::neural_network_type;

use crate::neural::network_f64::create_nullnetdata;

use std::io::{self, prelude::*, BufReader};

use ndarray;
use ndarray_npy;

pub fn largefile_to_hash_cpu(
    filename: &str,
    batch_size: u64,
) -> nohash_hasher::IntMap<u64, Vec<f64>> {
    let mut lookup: nohash_hasher::IntMap<u64, Vec<f64>> = nohash_hasher::IntMap::default();

    let nparr: ndarray::Array2<f64> = ndarray_npy::read_npy(filename).unwrap();

    let numrows = nparr.dim().0 as u64;

    //println!("{:?}",arr.row(6).to_vec());

    let mut tempbatch: Vec<f64> = Vec::new();

    let mut idx = 0;

    let mut batch_idx = 0;

    for qq in 0..numrows {
        let current_vec = nparr.row(qq as usize).to_vec();
        tempbatch.extend(current_vec);
        batch_idx = batch_idx + 1;

        if batch_idx >= batch_size {
            lookup.insert(idx, tempbatch.clone());
            tempbatch = Vec::new();
            batch_idx = 0;
            idx = idx + 1;
        }
    }

    lookup
}
