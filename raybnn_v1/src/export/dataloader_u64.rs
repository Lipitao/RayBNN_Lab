extern crate arrayfire;

use std::fs;

use rayon::array;
use rayon::prelude::*;

pub fn find_model_paths(dir_path: &str) -> Vec<String> {
    let mut models: Vec<String> = Vec::new();

    let paths = fs::read_dir(dir_path).unwrap();

    for path in paths {
        let pathstr = path.unwrap().path().to_str().unwrap().to_string();

        if pathstr.contains("active_size") && pathstr.contains(".csv") {
            models.push(pathstr);
        }
    }

    models
}

pub fn find_cube_paths(dir_path: &str) -> Vec<String> {
    let mut models: Vec<String> = Vec::new();

    let paths = fs::read_dir(dir_path).unwrap();

    for path in paths {
        let pathstr = path.unwrap().path().to_str().unwrap().to_string();

        if pathstr.contains("cube") && pathstr.contains(".csv") {
            models.push(pathstr);
        }
    }

    models
}

pub fn extract_file_info(filepath: &str) -> Vec<u64> {
    let targetstr = filepath.clone().replace(".csv", "");

    let strsplit: Vec<&str> = targetstr.split('_').collect();

    let mut outdata: Vec<u64> = Vec::new();

    let mut idx = 0;

    let mut parse_state = 0;
    for tmp in strsplit {
        if tmp.contains("active") && (parse_state == 0) {
            parse_state = 1;
        }

        if (parse_state == 1) {
            if (idx % 3) == 2 {
                //println!("data {}",tmp);
                let elem = tmp.clone().parse::<u64>().unwrap();
                outdata.push(elem);
            }

            idx = idx + 1;
        }
    }

    outdata
}

pub fn str_to_vec(instr: &str) -> arrayfire::Array<u64> {
    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let mut outarr = arrayfire::constant::<u64>(0, temp_dims);

    let mut newline = instr.replace("\n", "");
    newline = newline.replace(" ", "");

    let strvec: Vec<&str> = newline.split(",").collect();
    let ssize: u64 = strvec.len() as u64;

    if (ssize > 1) {
        let mut vecu64: Vec<u64> = Vec::new();
        for i in 0u64..ssize {
            let value: u64 = strvec[i as usize].parse::<u64>().unwrap();
            vecu64.push(value);
        }

        let new_dims = arrayfire::Dim4::new(&[ssize, 1, 1, 1]);
        outarr = arrayfire::Array::new(&vecu64, new_dims);
    }

    outarr
}

pub fn file_to_matrix(filename: &str, dims: arrayfire::Dim4) -> arrayfire::Array<u64> {
    let mut outarr = arrayfire::constant::<u64>(0, dims);
    let row_num: i64 = dims[0] as i64;

    let contents = fs::read_to_string(filename).expect("error");
    let mut lines = contents.split("\n");
    for i in 0i64..row_num {
        let row = str_to_vec(lines.next().unwrap());
        arrayfire::set_row(&mut outarr, &row, i);
    }

    outarr
}

pub fn vec_to_str(invec: &arrayfire::Array<u64>) -> String {
    let mut vec0 = vec![u64::default(); invec.elements()];
    invec.host(&mut vec0);
    let mut s0 = format!("{:?}", vec0);
    s0 = s0.replace("[", "");
    s0 = s0.replace("]", "");
    s0 = s0.replace(" ", "");

    s0
}

pub fn vec_cpu_to_str(invec: &Vec<u64>) -> String {
    let mut s0 = format!("{:?}", invec.clone());
    s0 = s0.replace("[", "");
    s0 = s0.replace("]", "");
    s0 = s0.replace(" ", "");

    s0
}

pub fn str_to_vec_cpu(instr: &str) -> Vec<u64> {
    let mut newline = instr.replace("\n", "");
    newline = newline.replace(" ", "");

    let strvec: Vec<&str> = newline.split(",").collect();
    let ssize: u64 = strvec.len() as u64;

    let mut vecu64: Vec<u64> = Vec::new();
    if (ssize > 1) {
        for i in 0u64..ssize {
            let value: u64 = strvec[i as usize].parse::<u64>().unwrap();
            vecu64.push(value);
        }
    }

    vecu64
}

pub fn file_to_vec_cpu(filename: &str) -> Vec<u64> {
    let contents = fs::read_to_string(filename).expect("error");

    contents
        .par_split('\n')
        .map(str_to_vec_cpu)
        .flatten_iter()
        .collect()
}
