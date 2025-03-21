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

pub fn vec_to_str(invec: &arrayfire::Array<f64>) -> String {
    let mut vec0 = vec![f64::default(); invec.elements()];
    invec.host(&mut vec0);
    let mut s0 = format!("{:?}", vec0);
    s0 = s0.replace("[", "");
    s0 = s0.replace("]", "");
    s0 = s0.replace(" ", "");

    s0
}

pub fn vec_cpu_to_str(invec: &Vec<f64>) -> String {
    let mut s0 = format!("{:?}", invec.clone());
    s0 = s0.replace("[", "");
    s0 = s0.replace("]", "");
    s0 = s0.replace(" ", "");

    s0
}

pub fn str_to_vec_cpu(instr: &str) -> Vec<f64> {
    let mut vecf64: Vec<f64> = Vec::new();

    let mut newline = instr.replace("\n", "");
    newline = newline.replace(" ", "");

    if newline.len() > 0 {
        let strvec: Vec<&str> = newline.split(",").collect();
        let ssize: u64 = strvec.len() as u64;

        for i in 0u64..ssize {
            let value: f64 = strvec[i as usize].parse::<f64>().unwrap();
            vecf64.push(value);
        }
    }

    vecf64
}

pub fn str_to_vec(instr: &str) -> arrayfire::Array<f64> {
    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let mut outarr = arrayfire::constant::<f64>(0.0, temp_dims);

    let mut newline = instr.replace("\n", "");
    newline = newline.replace(" ", "");

    let strvec: Vec<&str> = newline.split(",").collect();
    let ssize: u64 = strvec.len() as u64;

    if (ssize > 1) {
        let mut vecf64: Vec<f64> = Vec::new();
        for i in 0u64..ssize {
            let value: f64 = strvec[i as usize].parse::<f64>().unwrap();
            vecf64.push(value);
        }

        let new_dims = arrayfire::Dim4::new(&[ssize, 1, 1, 1]);
        outarr = arrayfire::Array::new(&vecf64, new_dims);
    }

    outarr
}

pub fn str_to_matrix(instr: &str, dims: arrayfire::Dim4) -> arrayfire::Array<f64> {
    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let mut outarr = arrayfire::constant::<f64>(0.0, temp_dims);

    let mut newline = instr.replace("\n", "");
    newline = newline.replace(" ", "");

    let strvec: Vec<&str> = newline.split(",").collect();
    let ssize: u64 = strvec.len() as u64;

    if (ssize > 1) {
        let mut vecf64: Vec<f64> = Vec::new();
        for i in 0u64..ssize {
            let value: f64 = strvec[i as usize].parse::<f64>().unwrap();
            vecf64.push(value);
        }

        outarr = arrayfire::Array::new(&vecf64, dims);
    }

    outarr
}

pub fn extract_file_info2(filepath: &str) -> Vec<f64> {
    let targetstr = filepath.clone().replace(".csv", "");

    let strsplit: Vec<&str> = targetstr.split('_').collect();

    let mut outdata: Vec<f64> = Vec::new();

    let mut idx = 0;

    let mut parse_state = 0;
    for tmp in strsplit {
        if tmp.contains("cube") && (parse_state == 0) {
            parse_state = 1;
        }

        if (parse_state == 1) {
            if idx == 1 {
                //println!("data {}",tmp);
                let elem = tmp.clone().parse::<f64>().unwrap();
                outdata.push(elem);
                break;
            }

            idx = idx + 1;
        }
    }

    outdata
}

pub fn file_to_vec_cpu(filename: &str) -> Vec<f64> {
    let contents = fs::read_to_string(filename).expect("error");

    contents
        .par_split('\n')
        .map(str_to_vec_cpu)
        .flatten_iter()
        .collect()
}

pub fn file_to_matrix(filename: &str, dims: arrayfire::Dim4) -> arrayfire::Array<f64> {
    let arr = file_to_vec_cpu(filename);

    let dims2 = arrayfire::Dim4::new(&[dims[1], dims[0], 1, 1]);
    let mut outarr = arrayfire::Array::new(&arr, dims2);

    arrayfire::transpose(&outarr, false)
}

pub fn file_to_hash_cpu(
    filename: &str,
    sample_size: u64,
    batch_size: u64,
) -> nohash_hasher::IntMap<u64, Vec<f64>> {
    let arr = file_to_vec_cpu(filename);

    let arr_size = arr.len() as u64;
    let item_num = (arr_size / (sample_size * batch_size));

    let mut lookup: nohash_hasher::IntMap<u64, Vec<f64>> = nohash_hasher::IntMap::default();
    let mut start: usize = 0;
    let mut end: usize = 0;
    for i in 0..item_num {
        start = (i * (sample_size * batch_size)) as usize;
        end = ((i + 1) * (sample_size * batch_size)) as usize;
        lookup.insert(i, (&arr[start..end]).to_vec());
    }

    lookup
}

pub fn compute_stats(
    input_size: u64,
    dataset: &nohash_hasher::IntMap<u64, Vec<f64>>,

    mean: &mut arrayfire::Array<f64>,
    stdev: &mut arrayfire::Array<f64>,
) {
    let mut tempvec = dataset[&0].clone();

    let mut veclen = tempvec.len() as u64;
    let mut formatarr = arrayfire::Array::new(
        &tempvec,
        arrayfire::Dim4::new(&[input_size, veclen / input_size, 1, 1]),
    );

    *mean = arrayfire::mean(&formatarr, 1);
    *stdev = arrayfire::stdev_v2(&formatarr, arrayfire::VarianceBias::SAMPLE, 1);

    for (key, value) in dataset {
        tempvec = value.clone();

        veclen = tempvec.len() as u64;
        formatarr = arrayfire::Array::new(
            &tempvec,
            arrayfire::Dim4::new(&[input_size, veclen / input_size, 1, 1]),
        );

        let tempmean = arrayfire::mean(&formatarr, 1);
        let tempstdev = arrayfire::stdev_v2(&formatarr, arrayfire::VarianceBias::SAMPLE, 1);

        *mean = (mean.clone() + tempmean) / 2.0f64;
        *stdev = (stdev.clone() + tempstdev) / 2.0f64;
    }

    let eps = 0.001f64;

    //  (eps > stdev )
    let CMP1 = arrayfire::gt(&eps, stdev, false);

    let selidx = arrayfire::locate(&CMP1);

    if selidx.dims()[0] > 0 {
        let mut ones = arrayfire::constant::<f64>(1.0, selidx.dims());

        let mut idxrs2 = arrayfire::Indexer::default();
        idxrs2.set_index(&selidx, 0, None);
        arrayfire::assign_gen(stdev, &idxrs2, &ones);
    }
}

pub fn normalize_dataset(
    input_size: u64,

    mean: &arrayfire::Array<f64>,
    stdev: &arrayfire::Array<f64>,

    dataset: &nohash_hasher::IntMap<u64, Vec<f64>>,
) -> nohash_hasher::IntMap<u64, Vec<f64>> {
    let mut tempdata = dataset.clone();

    for (key, value) in dataset {
        let tempvec = value.clone();

        let veclen = tempvec.len() as u64;
        let mut formatarr = arrayfire::Array::new(
            &tempvec,
            arrayfire::Dim4::new(&[input_size, veclen / input_size, 1, 1]),
        );

        //formatarr = (formatarr - mean)/stdev;
        formatarr = arrayfire::sub(&formatarr, mean, true);
        formatarr = arrayfire::div(&formatarr, stdev, true);

        let mut tempvec = vec![f64::default(); formatarr.elements()];
        formatarr.host(&mut tempvec);

        tempdata.insert(key.clone(), tempvec);
    }

    tempdata
}

pub fn largefile_to_hash_cpu(filename: &str) -> nohash_hasher::IntMap<u64, Vec<f64>> {
    let mut lookup: nohash_hasher::IntMap<u64, Vec<f64>> = nohash_hasher::IntMap::default();

    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    let mut idx = 0;
    for line in reader.lines() {
        let tempstr = line.unwrap().clone();
        let outvec = str_to_vec_cpu(&tempstr);
        lookup.insert(idx, outvec.clone());
        idx = idx + 1;
    }

    lookup
}

pub fn shuffle_hash_cpu(
    dataX: &mut nohash_hasher::IntMap<u64, Vec<f64>>,
    dataY: &mut nohash_hasher::IntMap<u64, Vec<f64>>,
) {
    let totalsize = dataX.keys().len() as u64;

    let randarr_dims = arrayfire::Dim4::new(&[totalsize, 1, 1, 1]);

    let randarr = arrayfire::randu::<f64>(randarr_dims);

    let (_, idx) = arrayfire::sort_index(&randarr, 0, false);

    let mut idxvec = vec![u32::default(); idx.elements()];
    idx.host(&mut idxvec);

    let mut newX: nohash_hasher::IntMap<u64, Vec<f64>> = nohash_hasher::IntMap::default();
    let mut newY: nohash_hasher::IntMap<u64, Vec<f64>> = nohash_hasher::IntMap::default();

    let mut newidx: u64 = 0;
    for i in 0..totalsize {
        newidx = idxvec[i as usize] as u64;

        newX.insert(newidx, (*dataX)[&i].clone());
        newY.insert(newidx, (*dataY)[&i].clone());
    }

    *dataX = newX;
    *dataY = newY;
}

pub fn format_vec_to_matrix(input: &Vec<f64>, dims: arrayfire::Dim4) -> arrayfire::Array<f64> {
    let dims2 = arrayfire::Dim4::new(&[dims[1], dims[0], 1, 1]);
    let outarr = arrayfire::Array::new(input, dims2);

    arrayfire::transpose(&outarr, false)
}

pub fn file_to_hash_matrix(
    filename: &str,
    sample_size: u64,
    batch_size: u64,
    dims: arrayfire::Dim4,
) -> nohash_hasher::IntMap<u64, arrayfire::Array<f64>> {
    let mut lookup2: nohash_hasher::IntMap<u64, arrayfire::Array<f64>> =
        nohash_hasher::IntMap::default();

    let lookup = file_to_hash_cpu(filename, sample_size, batch_size);

    let item_num = lookup.len() as u64;

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut temparr = arrayfire::constant::<f64>(0.0, temp_dims);
    for i in 0..item_num {
        temparr = arrayfire::Array::new(&lookup[&i], dims);

        lookup2.insert(i, temparr);
    }

    lookup2
}

pub fn write_arr_to_csv(filename: &str, arr: &arrayfire::Array<f64>) {
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

pub fn write_vec_cpu_to_csv(filename: &str, invec: &Vec<f64>) {
    let mut wtr0 = vec_cpu_to_str(invec);

    //wtr0 = wtr0.replace(",", "\n");

    let mut file0 = File::create(filename).unwrap();
    writeln!(file0, "{}", wtr0);
}

pub fn hash_batch_to_files(filename: &str, hash_map: &nohash_hasher::IntMap<u64, Vec<f64>>) {
    for batch_idx in 0..(hash_map.len() as u64) {
        let cur_batch = hash_map[&batch_idx].clone();

        let tmpfilename = format!("{}_batch_idx_{}.yhat", filename, batch_idx);

        write_vec_cpu_to_csv(&tmpfilename, &cur_batch);
    }
}

pub fn save_network(
    filename: &str,
    netdata: &network_metadata_type,
    WValues: &arrayfire::Array<f64>,
    WRowIdxCSR: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
    H: &arrayfire::Array<f64>,
    A: &arrayfire::Array<f64>,
    B: &arrayfire::Array<f64>,
    C: &arrayfire::Array<f64>,
    D: &arrayfire::Array<f64>,
    E: &arrayfire::Array<f64>,
    glia_pos: &arrayfire::Array<f64>,
    neuron_pos: &arrayfire::Array<f64>,
    neuron_idx: &arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();
    let batch_size: u64 = netdata.batch_size.clone();

    let del_unused_neuron: bool = netdata.del_unused_neuron.clone();

    let time_step: f64 = netdata.time_step.clone();
    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let init_prob: f64 = netdata.init_prob.clone();
    let add_neuron_rate: f64 = netdata.add_neuron_rate.clone();
    let del_neuron_rate: f64 = netdata.del_neuron_rate.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let mut wtr0: Vec<String> = Vec::new();

    let s0 = vec_to_str(WValues);
    wtr0.push(s0);

    let s0 = vec_to_str_i32(WRowIdxCSR);
    wtr0.push(s0);

    let s0 = vec_to_str_i32(WColIdx);
    wtr0.push(s0);

    let s0 = vec_to_str(H);
    wtr0.push(s0);

    let s0 = vec_to_str(A);
    wtr0.push(s0);

    let s0 = vec_to_str(B);
    wtr0.push(s0);

    let s0 = vec_to_str(C);
    wtr0.push(s0);

    let s0 = vec_to_str(D);
    wtr0.push(s0);

    let s0 = vec_to_str(E);
    wtr0.push(s0);

    let s0 = vec_to_str(glia_pos);
    wtr0.push(s0);

    let s0 = vec_to_str(neuron_pos);
    wtr0.push(s0);

    let s0 = vec_to_str_i32(neuron_idx);
    wtr0.push(s0);

    let netarr0: Vec<u64> = vec![
        neuron_size,
        input_size,
        output_size,
        proc_num,
        active_size,
        space_dims,
        step_num,
        batch_size,
        del_unused_neuron as u64,
    ];

    let s0 = vec_cpu_to_str_u64(&netarr0);
    wtr0.push(s0);

    let netarr1: Vec<f64> = vec![
        time_step,
        nratio,
        neuron_std,
        sphere_rad,
        neuron_rad,
        con_rad,
        init_prob,
        add_neuron_rate,
        del_neuron_rate,
        center_const,
        spring_const,
        repel_const,
    ];

    let s0 = vec_cpu_to_str(&netarr1);
    wtr0.push(s0);

    let mut file0 = File::create(filename).unwrap();
    writeln!(file0, "{}", wtr0.join("\n"));
}

pub fn save_network2(filename: &str, neural_network: &neural_network_type) {
    let WValuesdims0 = neural_network.WColIdx.dims()[0];

    let network_paramsdims0 = neural_network.network_params.dims()[0];

    let Hdims0 = (network_paramsdims0 - WValuesdims0) / 6;

    let Wstart = 0;
    let Wend = (WValuesdims0 as i64) - 1;

    let Hstart = Wend + 1;
    let Hend = Hstart + (Hdims0 as i64) - 1;

    let Astart = Hend + 1;
    let Aend = Astart + (Hdims0 as i64) - 1;

    let Bstart = Aend + 1;
    let Bend = Bstart + (Hdims0 as i64) - 1;

    let Cstart = Bend + 1;
    let Cend = Cstart + (Hdims0 as i64) - 1;

    let Dstart = Cend + 1;
    let Dend = Dstart + (Hdims0 as i64) - 1;

    let Estart = Dend + 1;
    let Eend = Estart + (Hdims0 as i64) - 1;

    let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
    let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
    let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
    let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
    let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
    let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
    let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];

    let WValues = arrayfire::index(&neural_network.network_params, &Wseqs);
    let H = arrayfire::index(&neural_network.network_params, &Hseqs);
    let A = arrayfire::index(&neural_network.network_params, &Aseqs);
    let B = arrayfire::index(&neural_network.network_params, &Bseqs);
    let C = arrayfire::index(&neural_network.network_params, &Cseqs);
    let D = arrayfire::index(&neural_network.network_params, &Dseqs);
    let E = arrayfire::index(&neural_network.network_params, &Eseqs);

    save_network(
        filename,
        &neural_network.netdata,
        &WValues,
        &neural_network.WRowIdxCSR,
        &neural_network.WColIdx,
        &H,
        &A,
        &B,
        &C,
        &D,
        &E,
        &neural_network.glia_pos,
        &neural_network.neuron_pos,
        &neural_network.neuron_idx,
    );
}

pub fn load_network(
    filename: &str,
    netdata: &mut network_metadata_type,
    WValues: &mut arrayfire::Array<f64>,
    WRowIdxCSR: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    H: &mut arrayfire::Array<f64>,
    A: &mut arrayfire::Array<f64>,
    B: &mut arrayfire::Array<f64>,
    C: &mut arrayfire::Array<f64>,
    D: &mut arrayfire::Array<f64>,
    E: &mut arrayfire::Array<f64>,
    glia_pos: &mut arrayfire::Array<f64>,
    neuron_pos: &mut arrayfire::Array<f64>,
    neuron_idx: &mut arrayfire::Array<i32>,
) {
    let mut netarr0: Vec<u64> = Vec::new();
    let mut netarr1: Vec<f64> = Vec::new();

    let contents = fs::read_to_string(filename).expect("error");
    let mut lines = contents.split("\n");
    let mut i = 0;
    for line in lines {
        match i {
            0 => *WValues = str_to_vec(&line),
            1 => *WRowIdxCSR = str_to_vec_i32(&line),
            2 => *WColIdx = str_to_vec_i32(&line),
            3 => *H = str_to_vec(&line),
            4 => *A = str_to_vec(&line),
            5 => *B = str_to_vec(&line),
            6 => *C = str_to_vec(&line),

            7 => *D = str_to_vec(&line),

            8 => *E = str_to_vec(&line),

            9 => *glia_pos = str_to_vec(&line),

            10 => *neuron_pos = str_to_vec(&line),

            11 => *neuron_idx = str_to_vec_i32(&line),

            12 => netarr0 = str_to_vec_cpu_u64(&line),

            13 => netarr1 = str_to_vec_cpu(&line),

            _ => println!("error"),
        }

        i = i + 1;
    }

    let neuron_size: u64 = netarr0[0];
    let input_size: u64 = netarr0[1];
    let output_size: u64 = netarr0[2];
    let proc_num: u64 = netarr0[3];
    let active_size: u64 = netarr0[4];
    let space_dims: u64 = netarr0[5];
    let step_num: u64 = netarr0[6];
    let batch_size: u64 = netarr0[7];

    let del_unused_neuron: bool = (netarr0[8] == 1);

    let time_step: f64 = netarr1[0];
    let nratio: f64 = netarr1[1];
    let neuron_std: f64 = netarr1[2];
    let sphere_rad: f64 = netarr1[3];
    let neuron_rad: f64 = netarr1[4];
    let con_rad: f64 = netarr1[5];
    let init_prob: f64 = netarr1[6];
    let add_neuron_rate: f64 = netarr1[7];
    let del_neuron_rate: f64 = netarr1[8];
    let center_const: f64 = netarr1[9];
    let spring_const: f64 = netarr1[10];
    let repel_const: f64 = netarr1[11];

    let newnetdata = network_metadata_type {
        neuron_size: neuron_size,
        input_size: input_size,
        output_size: output_size,
        proc_num: proc_num,
        active_size: active_size,
        space_dims: space_dims,
        step_num: step_num,
        batch_size: batch_size,
        del_unused_neuron: del_unused_neuron,

        time_step: time_step,
        nratio: nratio,
        neuron_std: neuron_std,
        sphere_rad: sphere_rad,
        neuron_rad: neuron_rad,
        con_rad: con_rad,
        init_prob: init_prob,
        add_neuron_rate: add_neuron_rate,
        del_neuron_rate: del_neuron_rate,
        center_const: center_const,
        spring_const: spring_const,
        repel_const: repel_const,
    };

    *netdata = newnetdata;

    let g_dims =
        arrayfire::Dim4::new(&[(glia_pos.dims()[0] / space_dims) as u64, space_dims, 1, 1]);
    *glia_pos = arrayfire::moddims(glia_pos, g_dims);

    let n_dims =
        arrayfire::Dim4::new(&[(neuron_pos.dims()[0] / space_dims) as u64, space_dims, 1, 1]);
    *neuron_pos = arrayfire::moddims(neuron_pos, n_dims);
}

pub fn load_network2(filename: &str) -> neural_network_type {
    let temp_dims = arrayfire::Dim4::new(&[4, 1, 1, 1]);

    let mut glia_pos = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut neuron_pos = arrayfire::constant::<f64>(0.0, temp_dims);

    let mut H = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut A = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut B = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut C = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut D = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut E = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut neuron_idx = arrayfire::constant::<i32>(0, temp_dims);

    let mut WValues = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut WRowIdxCSR = arrayfire::constant::<i32>(0, temp_dims);
    let mut WColIdx = arrayfire::constant::<i32>(0, temp_dims);

    let mut netdata = create_nullnetdata();

    load_network(
        filename,
        &mut netdata,
        &mut WValues,
        &mut WRowIdxCSR,
        &mut WColIdx,
        &mut H,
        &mut A,
        &mut B,
        &mut C,
        &mut D,
        &mut E,
        &mut glia_pos,
        &mut neuron_pos,
        &mut neuron_idx,
    );

    let total_param_size = WValues.dims()[0]
        + H.dims()[0]
        + A.dims()[0]
        + B.dims()[0]
        + C.dims()[0]
        + D.dims()[0]
        + E.dims()[0];
    let network_params_dims = arrayfire::Dim4::new(&[total_param_size, 1, 1, 1]);

    let Wstart = 0;
    let Wend = (WValues.dims()[0] as i64) - 1;

    let Hstart = Wend + 1;
    let Hend = Hstart + (H.dims()[0] as i64) - 1;

    let Astart = Hend + 1;
    let Aend = Astart + (A.dims()[0] as i64) - 1;

    let Bstart = Aend + 1;
    let Bend = Bstart + (B.dims()[0] as i64) - 1;

    let Cstart = Bend + 1;
    let Cend = Cstart + (C.dims()[0] as i64) - 1;

    let Dstart = Cend + 1;
    let Dend = Dstart + (D.dims()[0] as i64) - 1;

    let Estart = Dend + 1;
    let Eend = Estart + (E.dims()[0] as i64) - 1;

    let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
    let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
    let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
    let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
    let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
    let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
    let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];

    let mut network_params = arrayfire::constant::<f64>(0.0, network_params_dims);
    arrayfire::assign_seq(&mut network_params, &Wseqs, &WValues);
    arrayfire::assign_seq(&mut network_params, &Hseqs, &H);
    arrayfire::assign_seq(&mut network_params, &Aseqs, &A);
    arrayfire::assign_seq(&mut network_params, &Bseqs, &B);
    arrayfire::assign_seq(&mut network_params, &Cseqs, &C);
    arrayfire::assign_seq(&mut network_params, &Dseqs, &D);
    arrayfire::assign_seq(&mut network_params, &Eseqs, &E);

    let mut neural_network: neural_network_type = neural_network_type {
        netdata: netdata,
        WRowIdxCSR: WRowIdxCSR,
        WColIdx: WColIdx,
        network_params: network_params,
        glia_pos: glia_pos,
        neuron_pos: neuron_pos,
        neuron_idx: neuron_idx,
    };

    neural_network
}

pub fn load_network_structure(
    filename: &str,

    glia_pos: &mut arrayfire::Array<f64>,
    neuron_pos: &mut arrayfire::Array<f64>,
    neuron_idx: &mut arrayfire::Array<i32>,
) {
    let mut netarr0: Vec<u64> = Vec::new();
    let mut netarr1: Vec<f64> = Vec::new();

    let contents = fs::read_to_string(filename).expect("error");
    let mut lines = contents.split("\n");
    let mut i = 0;
    for line in lines {
        match i {
            0 => println!("error"),
            1 => println!("error"),
            2 => println!("error"),
            3 => println!("error"),
            4 => println!("error"),
            5 => println!("error"),
            6 => println!("error"),
            7 => println!("error"),
            8 => println!("error"),

            9 => *glia_pos = str_to_vec(&line),

            10 => *neuron_pos = str_to_vec(&line),

            11 => *neuron_idx = str_to_vec_i32(&line),

            12 => netarr0 = str_to_vec_cpu_u64(&line),

            13 => netarr1 = str_to_vec_cpu(&line),

            _ => println!("error"),
        }

        i = i + 1;
    }

    let neuron_size: u64 = netarr0[0];
    let input_size: u64 = netarr0[1];
    let output_size: u64 = netarr0[2];
    let proc_num: u64 = netarr0[3];
    let active_size: u64 = netarr0[4];
    let space_dims: u64 = netarr0[5];
    let step_num: u64 = netarr0[6];
    let batch_size: u64 = netarr0[7];

    let del_unused_neuron: bool = (netarr0[8] == 1);

    let time_step: f64 = netarr1[0];
    let nratio: f64 = netarr1[1];
    let neuron_std: f64 = netarr1[2];
    let sphere_rad: f64 = netarr1[3];
    let neuron_rad: f64 = netarr1[4];
    let con_rad: f64 = netarr1[5];
    let init_prob: f64 = netarr1[6];
    let add_neuron_rate: f64 = netarr1[7];
    let del_neuron_rate: f64 = netarr1[8];
    let center_const: f64 = netarr1[9];
    let spring_const: f64 = netarr1[10];
    let repel_const: f64 = netarr1[11];

    let newnetdata = network_metadata_type {
        neuron_size: neuron_size,
        input_size: input_size,
        output_size: output_size,
        proc_num: proc_num,
        active_size: active_size,
        space_dims: space_dims,
        step_num: step_num,
        batch_size: batch_size,
        del_unused_neuron: del_unused_neuron,

        time_step: time_step,
        nratio: nratio,
        neuron_std: neuron_std,
        sphere_rad: sphere_rad,
        neuron_rad: neuron_rad,
        con_rad: con_rad,
        init_prob: init_prob,
        add_neuron_rate: add_neuron_rate,
        del_neuron_rate: del_neuron_rate,
        center_const: center_const,
        spring_const: spring_const,
        repel_const: repel_const,
    };

    //*netdata =  newnetdata;

    let g_dims =
        arrayfire::Dim4::new(&[(glia_pos.dims()[0] / space_dims) as u64, space_dims, 1, 1]);
    *glia_pos = arrayfire::moddims(glia_pos, g_dims);

    let n_dims =
        arrayfire::Dim4::new(&[(neuron_pos.dims()[0] / space_dims) as u64, space_dims, 1, 1]);
    *neuron_pos = arrayfire::moddims(neuron_pos, n_dims);
}
