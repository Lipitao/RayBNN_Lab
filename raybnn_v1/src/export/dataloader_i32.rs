extern crate arrayfire;

use std::fs::File;
use std::io::Write;

use std::fs;

pub fn str_to_vec(instr: &str) -> arrayfire::Array<i32> {
    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let mut outarr = arrayfire::constant::<i32>(0, temp_dims);

    let mut newline = instr.replace("\n", "");
    newline = newline.replace(" ", "");

    let strvec: Vec<&str> = newline.split(",").collect();
    let ssize: u64 = strvec.len() as u64;

    if (ssize > 1) {
        let mut veci32: Vec<i32> = Vec::new();
        for i in 0u64..ssize {
            let value: i32 = strvec[i as usize].parse::<i32>().unwrap();
            veci32.push(value);
        }

        let new_dims = arrayfire::Dim4::new(&[ssize, 1, 1, 1]);
        outarr = arrayfire::Array::new(&veci32, new_dims);
    }

    outarr
}

pub fn file_to_matrix(filename: &str, dims: arrayfire::Dim4) -> arrayfire::Array<i32> {
    let mut outarr = arrayfire::constant::<i32>(0, dims);
    let row_num: i64 = dims[0] as i64;

    let contents = fs::read_to_string(filename).expect("error");
    let mut lines = contents.split("\n");
    for i in 0i64..row_num {
        let row = str_to_vec(lines.next().unwrap());
        arrayfire::set_row(&mut outarr, &row, i);
    }

    outarr
}

pub fn vec_to_str(invec: &arrayfire::Array<i32>) -> String {
    let mut vec0 = vec![i32::default(); invec.elements()];
    invec.host(&mut vec0);
    let mut s0 = format!("{:?}", vec0);
    s0 = s0.replace("[", "");
    s0 = s0.replace("]", "");
    s0 = s0.replace(" ", "");

    s0
}

pub fn vec_cpu_to_str(invec: &Vec<i32>) -> String {
    let mut s0 = format!("{:?}", invec.clone());
    s0 = s0.replace("[", "");
    s0 = s0.replace("]", "");
    s0 = s0.replace(" ", "");

    s0
}

pub fn write_arr_to_csv(filename: &str, arr: &arrayfire::Array<i32>) {
    let mut wtr0: Vec<String> = Vec::new();

    let item_num = arr.dims()[0] as i64;

    for i in 0..item_num {
        let cur_item = arrayfire::row(arr, i);
        let s0 = vec_to_str(&cur_item);
        wtr0.push(s0);
    }

    let mut file0 = File::create(filename).unwrap();
    writeln!(file0, "{}", wtr0.join("\n"));
}
