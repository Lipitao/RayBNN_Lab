extern crate arrayfire;
use nohash_hasher;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

use num::Float;
use num::ToPrimitive;

use crate::export::rand_f64::single_random_uniform;

use crate::neural::network_f64::network_metadata_type;
use crate::physics::update_f64::add_neuron_to_existing;
use crate::physics::update_f64::add_neuron_to_existing2;
use crate::physics::update_f64::reduce_network_size;

use crate::interpol::linear_f64::find;
use crate::physics::distance_f64::vec_min_dist;

use crate::export::dataloader_f64::save_network2;

use crate::export::dataloader_f64::load_network;
use crate::export::dataloader_f64::save_network;
use crate::graph::large_sparse_u64::COO_batch_find;

use crate::export::dataloader_f64::vec_cpu_to_str;
use crate::export::dataloader_u64::vec_cpu_to_str as vec_cpu_to_str_u64;

use crate::graph::adjacency_f64::delete_neurons_at_idx;
use crate::graph::adjacency_f64::delete_smallest_weights;
use crate::graph::adjacency_f64::delete_unused_neurons;

use crate::export::dataloader_f64::load_network_structure;

use crate::graph::large_sparse_i32::CSR_to_COO;

use crate::interface::automatic_f64::arch_search_type;

use crate::export::dataloader_u64::extract_file_info;

use crate::export::dataloader_u64::find_model_paths;

use crate::export::dataloader_u64::find_cube_paths;

use crate::export::dataloader_f64::extract_file_info2;

use crate::export::dataloader_f64::load_network2;

use crate::neural::network_f64::clone_netdata;

use crate::export::dataloader_f64::file_to_vec_cpu;

use crate::interface::automatic_f64::set_network_seed2;

use crate::export::rand_u64::random_uniform_range;

use serde::{Deserialize, Serialize};

const LARGE_POS_NUM_f64: f64 = 1.0e9;
const SMALL_POS_NUM_f64: f64 = 1.0e-6;

const LOWER_PROB: f64 = 0.3;
const HIGHER_PROB: f64 = 0.7;

const COO_find_limit: u64 = 1500000000;

const high: f64 = 1000000.0;
const low: f64 = -1000000.0;

const high_u64: u64 = 1000000;
const low_u64: u64 = 0;

#[derive(Serialize, Deserialize)]
pub enum evolution_search_type {
    TOP5_SEARCH,
    METROPOLIS_SEARCH,
    SURROGATE_SPLINE_SEARCH,
}

#[derive(Serialize, Deserialize)]
pub struct evolution_info_type {
    pub dir_path: String,
    pub cur_path: String,

    pub search_strategy: evolution_search_type,
    pub success_idx: u64,
    pub total_tries: u64,

    pub crossval_vec: Vec<f64>,
    pub netdata_vec: Vec<network_metadata_type>,

    pub traj_size: u64,

    pub max_input_size: u64,
    pub max_output_size: u64,

    pub max_active_size: u64,
    pub min_active_size: u64,

    pub max_proc_num: u64,
    pub min_proc_num: u64,

    pub max_proc_num_step: u64,
    pub min_proc_num_step: u64,

    pub max_active_size_step: f64,
    pub min_active_size_step: f64,

    pub max_prune_num: f64,
    pub min_prune_num: f64,

    pub max_search_num: u64,
}

#[derive(Serialize, Deserialize)]
pub struct evolution_prop {
    pub dir_name: String,

    pub max_active_size: u64,
    pub min_active_size: u64,

    pub max_proc_num: u64,
    pub min_proc_num: u64,

    pub max_proc_step: u64,
    pub min_proc_step: u64,

    pub max_search_num: u64,
    pub min_search_num: u64,

    pub max_prune_num: u64,
    pub min_prune_num: u64,
}

pub fn extract_info(
    netdata_vec: &Vec<network_metadata_type>,

    active_size_vec: &mut Vec<u64>,
    proc_num_vec: &mut Vec<u64>,
) {
    //Retrieve data
    let vecsize = netdata_vec.len();
    for i in 0..vecsize {
        active_size_vec.push(netdata_vec[i].active_size.clone());
        proc_num_vec.push(netdata_vec[i].proc_num.clone());
    }
}

pub fn sort_info(
    evolutiondata: &evolution_prop,
    crossval_vec: &Vec<f64>,
    active_size_vec: &Vec<u64>,
    proc_num_vec: &Vec<u64>,

    newcrossval_hash: &mut nohash_hasher::IntMap<u64, arrayfire::Array<f64>>,
    newactive_size_hash: &mut nohash_hasher::IntMap<u64, arrayfire::Array<u64>>,
    newproc_num_vec: &mut Vec<u64>,
) {
    *newcrossval_hash = nohash_hasher::IntMap::default();
    *newactive_size_hash = nohash_hasher::IntMap::default();
    *newproc_num_vec = Vec::new();

    let max_active_size = evolutiondata.max_active_size;
    let min_active_size = evolutiondata.min_active_size;

    let max_proc_num = evolutiondata.max_proc_num;
    let min_proc_num = evolutiondata.min_proc_num;

    let max_proc_step = evolutiondata.max_proc_step;
    let min_proc_step = evolutiondata.min_proc_step;

    let max_search_num = evolutiondata.max_search_num;
    let min_search_num = evolutiondata.min_search_num;

    let max_prune_num = evolutiondata.max_prune_num;
    let min_prune_num = evolutiondata.min_prune_num;

    let mut crossval_arr = arrayfire::Array::new(
        &crossval_vec,
        arrayfire::Dim4::new(&[crossval_vec.len() as u64, 1, 1, 1]),
    );

    let mut active_size_arr = arrayfire::Array::new(
        &active_size_vec,
        arrayfire::Dim4::new(&[active_size_vec.len() as u64, 1, 1, 1]),
    );

    let mut proc_num_arr = arrayfire::Array::new(
        &proc_num_vec,
        arrayfire::Dim4::new(&[proc_num_vec.len() as u64, 1, 1, 1]),
    );

    let gidx = (proc_num_arr.clone() * (max_active_size)) + active_size_arr.clone();

    let (_, idx) = arrayfire::sort_index(&gidx, 0, true);

    crossval_arr = arrayfire::lookup(&crossval_arr, &idx, 0);
    active_size_arr = arrayfire::lookup(&active_size_arr, &idx, 0);
    proc_num_arr = arrayfire::lookup(&proc_num_arr, &idx, 0);

    let mut crossval_vec_cpu = vec![f64::default(); crossval_arr.elements()];
    crossval_arr.host(&mut crossval_vec_cpu);

    let mut active_size_vec_cpu = vec![u64::default(); active_size_arr.elements()];
    active_size_arr.host(&mut active_size_vec_cpu);

    let mut proc_num_vec_cpu = vec![u64::default(); proc_num_arr.elements()];
    proc_num_arr.host(&mut proc_num_vec_cpu);

    //newcrossval_vec.push(Vec::new());
    //newactive_size_vec.push(Vec::new());

    //let mut j: usize = 0;
    let vecsize = crossval_vec_cpu.len();
    let mut cur_proc_num = proc_num_vec_cpu[0].clone();
    newproc_num_vec.push(cur_proc_num);

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let mut cur_crossval = arrayfire::constant::<f64>(crossval_vec_cpu[0].clone(), temp_dims);
    let mut cur_active_size = arrayfire::constant::<u64>(active_size_vec_cpu[0].clone(), temp_dims);

    let mut crossval_arr = cur_crossval.clone();
    let mut active_size_arr = cur_active_size.clone();

    for i in 1..vecsize {
        if (cur_proc_num != proc_num_vec_cpu[i]) {
            (*newcrossval_hash).insert(cur_proc_num, crossval_arr.clone());
            (*newactive_size_hash).insert(cur_proc_num, active_size_arr.clone());

            crossval_arr = arrayfire::constant::<f64>(crossval_vec_cpu[i].clone(), temp_dims);
            active_size_arr = arrayfire::constant::<u64>(active_size_vec_cpu[i].clone(), temp_dims);

            cur_proc_num = proc_num_vec_cpu[i].clone();
            newproc_num_vec.push(cur_proc_num);

            //newcrossval_vec.push(Vec::new());
            //newactive_size_vec.push(Vec::new());

            //j = j + 1;
        } else {
            //newcrossval_vec[j].push( crossval_vec_cpu[i].clone());
            //newactive_size_vec[j].push( active_size_vec_cpu[i].clone());

            cur_crossval = arrayfire::constant::<f64>(crossval_vec_cpu[i].clone(), temp_dims);
            cur_active_size = arrayfire::constant::<u64>(active_size_vec_cpu[i].clone(), temp_dims);

            crossval_arr = arrayfire::join(0, &crossval_arr, &cur_crossval);
            active_size_arr = arrayfire::join(0, &active_size_arr, &cur_active_size);
        }
    }

    (*newcrossval_hash).insert(cur_proc_num, crossval_arr.clone());
    (*newactive_size_hash).insert(cur_proc_num, active_size_arr.clone());
}

pub fn linear_sampling(
    evolutiondata: &evolution_prop,
    crossval_vec: &arrayfire::Array<f64>,
    active_size_vec: &arrayfire::Array<u64>,

    out_crossval_vec: &mut arrayfire::Array<f64>,
    out_active_size_vec: &mut arrayfire::Array<u64>,
) {
    let max_active_size = evolutiondata.max_active_size;
    let min_active_size = evolutiondata.min_active_size;

    let max_proc_num = evolutiondata.max_proc_num;
    let min_proc_num = evolutiondata.min_proc_num;

    let max_proc_step = evolutiondata.max_proc_step;
    let min_proc_step = evolutiondata.min_proc_step;

    let max_search_num = evolutiondata.max_search_num;
    let min_search_num = evolutiondata.min_search_num;

    let max_prune_num = evolutiondata.max_prune_num;
    let min_prune_num = evolutiondata.min_prune_num;

    let neuron_step = max_active_size / max_search_num;

    let sample_num: u64 = 5000;

    let active_size_f64 = active_size_vec.cast::<f64>();
    let (_, _, idx) = arrayfire::imin_all(crossval_vec);

    let mut active_size_cpu = vec![f64::default(); active_size_f64.elements()];
    active_size_f64.host(&mut active_size_cpu);

    let mut samples = arrayfire::randn::<f64>(arrayfire::Dim4::new(&[sample_num, 1, 1, 1]));
    samples = samples * neuron_step;
    samples = active_size_cpu[idx as usize] + samples;

    //let cmp1 = (samples < max_active_size);
    let cmp1 = arrayfire::lt(&samples, &max_active_size, false);
    //let cmp2 = (min_active_size < samples);
    let cmp2 = arrayfire::lt(&min_active_size, &samples, false);
    //let select = af::where( (cmp1*cmp2) == 1);
    let and1 = arrayfire::and(&cmp1, &cmp2, false);
    let idxvec = arrayfire::locate(&and1);

    samples = arrayfire::lookup(&samples, &idxvec, 0);

    let dc = arrayfire::diff1(crossval_vec, 0);
    let dn = (1.0E-5 as f64) + arrayfire::diff1(&active_size_f64, 0);

    let mut dcdn = dc / dn;

    //let firstelem = arrayfire::row(&dcdn,0);
    let lastelem = arrayfire::row(&dcdn, (dcdn.dims()[0] - 1) as i64);

    //dcdn = arrayfire::join(0, &firstelem, &dcdn);
    dcdn = arrayfire::join(0, &dcdn, &lastelem);

    let mut pred_cv = find(&active_size_f64, crossval_vec, &dcdn, &samples);

    let (_, idxvec2) = arrayfire::sort_index(&pred_cv, 0, true);

    samples = arrayfire::lookup(&samples, &idxvec2, 0);

    pred_cv = arrayfire::lookup(&pred_cv, &idxvec2, 0);

    let mut samples_u64 = samples.cast::<u64>();

    *out_crossval_vec = arrayfire::row(&pred_cv, 0);
    *out_active_size_vec = arrayfire::row(&samples_u64, 0);
    let mut out_active_size_vec_f64 = out_active_size_vec.clone().cast::<f64>();

    let vecsize = samples_u64.dims()[0] as i64;

    let neuron_step_f64 = neuron_step as f64;
    for i in 1..vecsize {
        let curitem = arrayfire::row(&samples_u64, i);
        let curitem_f64 = curitem.cast::<f64>();

        let curpred_cv = arrayfire::row(&pred_cv, i);

        if (vec_min_dist(&curitem_f64, &out_active_size_vec_f64).sqrt() > neuron_step_f64) {
            *out_active_size_vec = arrayfire::join(0, out_active_size_vec, &curitem);
            out_active_size_vec_f64 = arrayfire::join(0, &out_active_size_vec_f64, &curitem_f64);

            *out_crossval_vec = arrayfire::join(0, out_crossval_vec, &curpred_cv);
        }
    }

    *out_active_size_vec = arrayfire::clamp(out_active_size_vec, &low_u64, &high_u64, false);

    *out_crossval_vec = arrayfire::clamp(out_crossval_vec, &low, &high, false);
}

pub fn execute_sampling(
    evolutiondata: &evolution_prop,

    sample_func: impl Fn(
        &evolution_prop,
        &arrayfire::Array<f64>,
        &arrayfire::Array<u64>,

        &mut arrayfire::Array<f64>,
        &mut arrayfire::Array<u64>,
    ),

    newcrossval_hash: &nohash_hasher::IntMap<u64, arrayfire::Array<f64>>,
    newactive_size_hash: &nohash_hasher::IntMap<u64, arrayfire::Array<u64>>,
    newproc_num_vec: &Vec<u64>,

    out_crossval_arr: &mut arrayfire::Array<f64>,
    out_active_size_arr: &mut arrayfire::Array<u64>,
    out_proc_num_arr: &mut arrayfire::Array<u64>,
) {
    //Fit spline to data and find next sampling point
    let vecsize = newproc_num_vec.len() as usize;

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let mut cur_proc_num = newproc_num_vec[0].clone();

    //let mut cur_crossval = newcrossval_hash[&cur_proc_num].clone();
    let mut crossvalarr = newcrossval_hash[&cur_proc_num].clone();

    //let mut cur_active_size = newactive_size_hash[&cur_proc_num].clone();
    let mut active_sizearr = newactive_size_hash[&cur_proc_num].clone();

    //let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    //let mut out_crossval_arr = arrayfire::constant::<f64>(0.0,temp_dims);
    //let mut out_active_size_arr = arrayfire::constant::<u64>(0,temp_dims);
    //let mut out_proc_num_arr = arrayfire::constant::<u64>(0,temp_dims);

    let mut out_crossval = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut out_active_size = arrayfire::constant::<u64>(0, temp_dims);

    sample_func(
        evolutiondata,
        &crossvalarr,
        &active_sizearr,
        out_crossval_arr,
        out_active_size_arr,
    );

    let mut out_proc_num = arrayfire::constant::<u64>(cur_proc_num, out_active_size_arr.dims());

    *out_proc_num_arr = out_proc_num.clone();

    for i in 1..vecsize {
        cur_proc_num = newproc_num_vec[i].clone();

        //cur_crossval = newcrossval_hash[&cur_proc_num].clone();
        crossvalarr = newcrossval_hash[&cur_proc_num].clone();

        if (crossvalarr.dims()[0] < 3) {
            continue;
        }

        //cur_active_size = newactive_size_hash[&cur_proc_num].clone();
        active_sizearr = newactive_size_hash[&cur_proc_num].clone();

        sample_func(
            evolutiondata,
            &crossvalarr,
            &active_sizearr,
            &mut out_crossval,
            &mut out_active_size,
        );

        out_proc_num = arrayfire::constant::<u64>(cur_proc_num, out_active_size.dims());
        *out_proc_num_arr = arrayfire::join(0, out_proc_num_arr, &out_proc_num);

        *out_crossval_arr = arrayfire::join(0, out_crossval_arr, &out_crossval);
        *out_active_size_arr = arrayfire::join(0, out_active_size_arr, &out_active_size);
    }
}

pub fn filter_samples(
    evolutiondata: &evolution_prop,

    active_size_arrz: &arrayfire::Array<u64>,
    proc_num_arrz: &arrayfire::Array<u64>,

    out_active_size_arr: &mut arrayfire::Array<u64>,
    out_proc_num_arr: &mut arrayfire::Array<u64>,
) {
    let max_active_size = evolutiondata.max_active_size;
    let min_active_size = evolutiondata.min_active_size;

    //Filter out existing
    if out_proc_num_arr.dims()[0] > 50 {
        *out_proc_num_arr = arrayfire::rows(out_proc_num_arr, 0, 49);
        *out_active_size_arr = arrayfire::rows(out_active_size_arr, 0, 49);
    }

    let COO_batch_size = 1 + ((COO_find_limit / active_size_arrz.dims()[0]) as u64);

    //Compute global index
    let gidx1 = proc_num_arrz.clone() * (max_active_size) + active_size_arrz.clone();

    let gidx2 = out_proc_num_arr.clone() * (max_active_size) + out_active_size_arr.clone();

    let gidx3 = COO_batch_find(&gidx2, &gidx1, COO_batch_size);

    //Filter out existing connections
    if gidx3.dims()[0] > 0 {
        let mut table = arrayfire::constant::<bool>(true, gidx2.dims());

        let inarr = arrayfire::constant::<bool>(false, gidx3.dims());

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&gidx3, 0, None);
        arrayfire::assign_gen(&mut table, &idxrs, &inarr);

        let tempidx = arrayfire::locate(&table);

        *out_proc_num_arr = arrayfire::lookup(out_proc_num_arr, &tempidx, 0);
        *out_active_size_arr = arrayfire::lookup(out_active_size_arr, &tempidx, 0);
    }
}

pub fn update(
    evolutiondata: &evolution_prop,
    crossval_vec: &mut Vec<f64>,
    netdata_vec: &mut Vec<network_metadata_type>,

    cube_pos: &arrayfire::Array<f64>,
    cube_radius: f64,

    netdata: &mut network_metadata_type,
    WValues: &mut arrayfire::Array<f64>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
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
    //let active_size: u64 = netdata.active_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let mut proc_num: u64 = netdata.proc_num.clone();
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

    let dir_name = evolutiondata.dir_name.clone();

    let max_active_size = evolutiondata.max_active_size;
    let min_active_size = evolutiondata.min_active_size;

    let max_proc_num = evolutiondata.max_proc_num;
    let min_proc_num = evolutiondata.min_proc_num;

    let max_proc_step = evolutiondata.max_proc_step;
    let min_proc_step = evolutiondata.min_proc_step;

    let max_search_num = evolutiondata.max_search_num;
    let min_search_num = evolutiondata.min_search_num;

    let max_prune_num = evolutiondata.max_prune_num;
    let min_prune_num = evolutiondata.min_prune_num;

    //Save current network
    let mut active_size = neuron_idx.dims()[0];
    let mut hidden_size = active_size - output_size - input_size;
    (*netdata).active_size = active_size;

    let filename = format!(
        "{}/active_size_{}_proc_num_{}.csv",
        dir_name, active_size, proc_num
    );

    save_network(
        &filename, netdata, WValues, WRowIdxCSR, WColIdx, H, A, B, C, D, E, glia_pos, neuron_pos,
        neuron_idx,
    );

    //Add more neurons
    if crossval_vec.len() < 4 {
        let new_neuron_num = ((hidden_size as f64) * add_neuron_rate) as u64 - hidden_size;

        add_neuron_to_existing(
            cube_pos,
            cube_radius,
            new_neuron_num,
            netdata,
            WValues,
            WRowIdxCOO,
            WColIdx,
            neuron_pos,
            neuron_idx,
        );

        return;
    }

    //Get network information
    let mut active_size_vec: Vec<u64> = Vec::new();
    let mut proc_num_vec: Vec<u64> = Vec::new();

    extract_info(netdata_vec, &mut active_size_vec, &mut proc_num_vec);

    //Save Info
    let mut saveInfo: Vec<String> = Vec::new();

    let s0 = vec_cpu_to_str(crossval_vec);
    saveInfo.push(s0);

    let s0 = vec_cpu_to_str_u64(&active_size_vec);
    saveInfo.push(s0);

    let s0 = vec_cpu_to_str_u64(&proc_num_vec);
    saveInfo.push(s0);

    let filename2 = format!("{}/info.csv", dir_name);

    let mut fileInfo = File::create(filename2).unwrap();
    writeln!(fileInfo, "{}", saveInfo.join("\n"));

    //Sort network in to vec
    let mut newcrossval_hash: nohash_hasher::IntMap<u64, arrayfire::Array<f64>> =
        nohash_hasher::IntMap::default();
    let mut newactive_size_hash: nohash_hasher::IntMap<u64, arrayfire::Array<u64>> =
        nohash_hasher::IntMap::default();
    let mut newproc_num_vec: Vec<u64> = Vec::new();

    sort_info(
        evolutiondata,
        crossval_vec,
        &active_size_vec,
        &proc_num_vec,
        &mut newcrossval_hash,
        &mut newactive_size_hash,
        &mut newproc_num_vec,
    );

    //Sample on all proc num
    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let mut out_crossval_arr = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut out_active_size_arr = arrayfire::constant::<u64>(0, temp_dims);
    let mut out_proc_num_arr = arrayfire::constant::<u64>(0, temp_dims);

    execute_sampling(
        &evolutiondata,
        linear_sampling,
        &newcrossval_hash,
        &newactive_size_hash,
        &newproc_num_vec,
        &mut out_crossval_arr,
        &mut out_active_size_arr,
        &mut out_proc_num_arr,
    );

    //Find closest network
    let (_, idxvec) = arrayfire::sort_index(&out_crossval_arr, 0, true);

    //out_crossval_arr = arrayfire::lookup(&out_crossval_arr, &idxvec, 0);
    out_proc_num_arr = arrayfire::lookup(&out_proc_num_arr, &idxvec, 0);
    out_active_size_arr = arrayfire::lookup(&out_active_size_arr, &idxvec, 0);

    //Filter samples
    let active_size_arrz = arrayfire::Array::new(
        &active_size_vec,
        arrayfire::Dim4::new(&[active_size_vec.len() as u64, 1, 1, 1]),
    );
    let proc_num_arrz = arrayfire::Array::new(
        &proc_num_vec,
        arrayfire::Dim4::new(&[proc_num_vec.len() as u64, 1, 1, 1]),
    );

    filter_samples(
        &evolutiondata,
        &active_size_arrz,
        &proc_num_arrz,
        &mut out_active_size_arr,
        &mut out_proc_num_arr,
    );

    //Select the best
    //out_crossval_arr = arrayfire::row(&out_crossval_arr,  0);
    out_proc_num_arr = arrayfire::row(&out_proc_num_arr, 0);
    out_active_size_arr = arrayfire::row(&out_active_size_arr, 0);

    //Convert to CPU
    let mut next_proc_num_vec = vec![u64::default(); out_proc_num_arr.elements()];
    out_proc_num_arr.host(&mut next_proc_num_vec);

    proc_num = next_proc_num_vec[0];

    //Convert to CPU
    let mut next_active_size_vec = vec![u64::default(); out_active_size_arr.elements()];
    out_active_size_arr.host(&mut next_active_size_vec);

    let mut next_active_size = next_active_size_vec[0];

    //Compute dist to nearest model
    let mut sel_active_size = newactive_size_hash[&proc_num].clone();

    //sel_active_size.retain(|&x| x <  next_active_size);

    let sel_active_size_arr_f64 = sel_active_size.cast::<f64>();
    let out_active_size_arr_f64 = out_active_size_arr.cast::<f64>();

    let mut dist = arrayfire::sub(&sel_active_size_arr_f64, &out_active_size_arr_f64, true);

    dist = arrayfire::abs(&dist);

    let (_, _, idx) = arrayfire::imin_all(&dist);

    let mut sel_active_size_vec = vec![u64::default(); sel_active_size.elements()];
    sel_active_size.host(&mut sel_active_size_vec);

    //Load nearest network
    active_size = sel_active_size_vec[idx as usize].clone();
    (*netdata).active_size = active_size;
    (*netdata).proc_num = proc_num;

    let filename = format!(
        "{}/active_size_{}_proc_num_{}.csv",
        dir_name, active_size, proc_num
    );

    load_network(
        &filename, netdata, WValues, WRowIdxCSR, WColIdx, H, A, B, C, D, E, glia_pos, neuron_pos,
        neuron_idx,
    );

    *WRowIdxCOO = CSR_to_COO(WRowIdxCSR);

    let neuron_step = max_active_size / max_search_num;

    /*
    let rounds = ((crossval_vec.len() as u64) / min_search_num);

    if (rounds > 0) && ((rounds + min_proc_num) <= max_proc_num)
    {
        (*netdata).proc_num = min_proc_num+rounds;
        proc_num = (*netdata).proc_num;
    }
    */

    if (next_active_size > active_size) {
        //Add neurons
        let mut new_neuron_num = 2 * (next_active_size - active_size);

        if next_active_size >= (max_active_size - 1) {
            new_neuron_num = 2 * (max_active_size - active_size);
        }

        if new_neuron_num > (4 * neuron_step) {
            new_neuron_num = (4 * neuron_step);
        }

        let del_num = ((active_size as f64) * 0.01) as u64;

        delete_smallest_weights(WValues, WRowIdxCOO, WColIdx, del_num);

        delete_unused_neurons(
            netdata, WValues, WRowIdxCOO, WColIdx, glia_pos, neuron_pos, neuron_idx,
        );

        add_neuron_to_existing(
            cube_pos,
            cube_radius,
            new_neuron_num,
            netdata,
            WValues,
            WRowIdxCOO,
            WColIdx,
            neuron_pos,
            neuron_idx,
        );
    } else if (next_active_size < active_size) {
        let mut del_neuron_num = (active_size - next_active_size) as usize;

        let mut delvec: Vec<u64> = Vec::new();

        let mut curidx = 0;
        while (delvec.len() < del_neuron_num) {
            curidx = random_uniform_range(active_size);

            if (input_size <= curidx) && (curidx < (active_size - output_size)) {
                if (delvec.contains(&curidx) == false) {
                    delvec.push(curidx);
                }
            }
        }

        let delete_idx = arrayfire::Array::new(
            &delvec,
            arrayfire::Dim4::new(&[delvec.len() as u64, 1, 1, 1]),
        );

        let delete_idx2 = arrayfire::sort(&delete_idx, 0, true).cast::<i32>();

        delete_neurons_at_idx(&delete_idx2, WValues, WRowIdxCOO, WColIdx);

        delete_unused_neurons(
            netdata, WValues, WRowIdxCOO, WColIdx, glia_pos, neuron_pos, neuron_idx,
        );
    }
}

pub fn get_crossval_data(
    arch_search: &mut arch_search_type,

    checked_paths: &mut Vec<String>,
    max_success_idx: &mut u64,
) {
    //Find all model paths
    let model_path_vec3 = find_model_paths(&arch_search.evolution_info.dir_path);

    let mut model_path_vec2 = Vec::new();
    for path in model_path_vec3 {
        let info_vec = extract_file_info(&path);
        if info_vec.len() == 4 {
            model_path_vec2.push(path.clone());
        }
    }

    println!("model_path_vec2 {:?}", model_path_vec2.clone());

    //Sort all models
    let mut unordered_vec: Vec<u64> = Vec::new();

    for path in model_path_vec2.clone() {
        let info_vec = extract_file_info(&path);
        unordered_vec.push(info_vec[3].clone());
    }

    println!("unordered_vec {:?}", unordered_vec);
    let unordered_arr = arrayfire::Array::new(
        &unordered_vec,
        arrayfire::Dim4::new(&[unordered_vec.len() as u64, 1, 1, 1]),
    );

    let (_, idx) = arrayfire::sort_index(&unordered_arr, 0, true);

    let mut idx_cpu = vec![u32::default(); idx.elements()];
    idx.host(&mut idx_cpu);

    let mut model_path_vec: Vec<String> = Vec::new();
    for qq in idx_cpu {
        let item = model_path_vec2[qq as usize].clone();
        model_path_vec.push(item);
    }

    println!("model_path_vec {:?}", model_path_vec);

    //Reset info
    (*arch_search).evolution_info.crossval_vec = Vec::new();

    (*arch_search).evolution_info.netdata_vec = Vec::new();

    //let mut checked_paths: Vec<String> = Vec::new();
    checked_paths.clear();

    *max_success_idx = 0;

    //Search all models
    for path in model_path_vec {
        let info_vec = extract_file_info(&path);

        if info_vec.len() == 4 {
            println!("info_vec {:?}", info_vec);

            println!("path {}", path);

            if info_vec[3].clone() > *max_success_idx {
                *max_success_idx = info_vec[3].clone();
            }

            //Load neural network file
            let neural_network = load_network2(&path);

            let metadata = clone_netdata(&(neural_network.netdata));
            drop(neural_network);

            //Append metadata
            (*arch_search).evolution_info.netdata_vec.push(metadata);

            //Add to path vector
            checked_paths.push(path.clone());

            //Load metric
            let rootname = path.clone().replace(".csv", "");

            let targetpath = format!("{}.metric", rootname);

            let eval_metric_out = file_to_vec_cpu(&targetpath);

            //Take average
            let mut avgelem = eval_metric_out.iter().sum::<f64>() / (eval_metric_out.len() as f64);

            if avgelem.is_infinite() || avgelem.is_nan() {
                avgelem = LARGE_POS_NUM_f64;
            }

            println!("avgelem {}", avgelem.clone());

            //Append crossval
            (*arch_search).evolution_info.crossval_vec.push(avgelem);
        }
    }
}

pub fn search_top5(arch_search: &mut arch_search_type, checked_paths: &mut Vec<String>) -> usize {
    let mut crossval_arr = arrayfire::Array::new(
        &(*arch_search).evolution_info.crossval_vec,
        arrayfire::Dim4::new(&[
            (*arch_search).evolution_info.crossval_vec.len() as u64,
            1,
            1,
            1,
        ]),
    );

    //Get lowest cross val
    let (_, crossidx) = arrayfire::sort_index(&crossval_arr, 0, true);

    let mut crossidx_cpu = vec![u32::default(); crossidx.elements()];
    crossidx.host(&mut crossidx_cpu);

    let mut tempcrossval_vec = Vec::new();

    let mut tempnetdata_vec = Vec::new();

    let mut tempchecked_paths: Vec<String> = Vec::new();

    //Enumerate across lowest crossval to get network data
    for qq in 0..crossidx_cpu.len().min(5) {
        let tempidx = crossidx_cpu[qq];

        tempcrossval_vec.push((*arch_search).evolution_info.crossval_vec[tempidx as usize].clone());

        let tempnetdata =
            clone_netdata(&((*arch_search).evolution_info.netdata_vec[tempidx as usize]));

        tempnetdata_vec.push(tempnetdata);

        tempchecked_paths.push(checked_paths[tempidx as usize].clone());
    }

    //Get lowest 5 values
    (*arch_search).evolution_info.crossval_vec = tempcrossval_vec.clone();
    crossval_arr = arrayfire::Array::new(
        &(*arch_search).evolution_info.crossval_vec,
        arrayfire::Dim4::new(&[
            (*arch_search).evolution_info.crossval_vec.len() as u64,
            1,
            1,
            1,
        ]),
    );

    (*arch_search).evolution_info.netdata_vec = tempnetdata_vec;

    *checked_paths = tempchecked_paths.clone();

    arrayfire::print_gen("crossval_arr".to_string(), &crossval_arr, Some(6));

    //Generate selection probabilities
    let (min_val, _) = arrayfire::min_all(&crossval_arr);
    crossval_arr = crossval_arr + min_val + SMALL_POS_NUM_f64;

    let (max_val, _) = arrayfire::max_all(&crossval_arr);

    crossval_arr = crossval_arr / (max_val + SMALL_POS_NUM_f64);

    let mut prob = -crossval_arr;
    prob = arrayfire::exp(&prob);

    let (probsum, _) = arrayfire::sum_all(&prob);
    prob = prob.clone() / probsum;

    //Select next point
    let mut quit = false;
    let mut selected_idx = 0;
    while 1 == 1 {
        let randidx = arrayfire::randu::<f64>(prob.dims());
        let randarr = arrayfire::randu::<f64>(prob.dims());

        let (_, probidx) = arrayfire::sort_index(&randidx, 0, false);

        let mut probidx_cpu = vec![u32::default(); probidx.elements()];
        probidx.host(&mut probidx_cpu);

        let mut randarr_cpu = vec![f64::default(); randarr.elements()];
        randarr.host(&mut randarr_cpu);

        let mut prob_cpu = vec![f64::default(); prob.elements()];
        prob.host(&mut prob_cpu);

        for q in 0..(prob_cpu.len()) {
            let cur_index = probidx_cpu[q].clone() as usize;

            if prob_cpu[cur_index] > randarr_cpu[cur_index] {
                selected_idx = q.clone();
                quit = true;
                break;
            }
        }

        if quit == true {
            break;
        }
    }

    selected_idx
}

pub fn search_metropolis(
    arch_search: &mut arch_search_type,

    checked_paths: &mut Vec<String>,
    max_success_idx: u64,
) -> usize {
    let crossval_arr = arrayfire::Array::new(
        &(*arch_search).evolution_info.crossval_vec,
        arrayfire::Dim4::new(&[
            (*arch_search).evolution_info.crossval_vec.len() as u64,
            1,
            1,
            1,
        ]),
    );

    let (mut min_val, _) = arrayfire::min_all(&crossval_arr);
    min_val = min_val.abs() + SMALL_POS_NUM_f64;

    let mut selected_idx = 0;

    let cur_idx = checked_paths.len() - 1;
    let cur_crossval = (*arch_search).evolution_info.crossval_vec[cur_idx].clone();
    let cur_path = checked_paths[cur_idx].clone();

    let prev_idx = checked_paths.len() - 2;
    let prev_crossval = (*arch_search).evolution_info.crossval_vec[prev_idx].clone();

    println!("prev_crossval {}", prev_crossval);
    println!("cur_crossval {}", cur_crossval);

    if prev_crossval > cur_crossval {
        selected_idx = cur_idx as usize;
        println!("accept");
    } else {
        let diff = (prev_crossval - cur_crossval) / (min_val as f64);
        let expval = diff.exp();

        println!("diff {}", diff);
        println!("expval {}", expval);

        let rand_number = single_random_uniform();

        if (expval > rand_number) {
            println!("current");

            selected_idx = cur_idx as usize;
        } else {
            println!("cur_path {}", cur_path.clone());

            selected_idx = prev_idx as usize;
            std::fs::remove_file(cur_path);
        }
    }

    selected_idx
}

pub fn evolve_network(arch_search: &mut arch_search_type) {
    //Save total number of function calls
    let total_tries = (*arch_search).evolution_info.total_tries;
    (*arch_search).evolution_info.total_tries = total_tries + 1;

    //Find all cube paths
    let cube_path_vec = find_cube_paths(&arch_search.evolution_info.dir_path);

    println!("cube_path_vec {:?}", cube_path_vec);

    let cube_path = cube_path_vec[0].clone();

    let cube_radius = extract_file_info2(&cube_path)[0];

    println!("cube_radius {}", cube_radius);

    let mut checked_paths: Vec<String> = Vec::new();
    let mut max_success_idx = 0;

    // Get cross validation data
    get_crossval_data(arch_search, &mut checked_paths, &mut max_success_idx);
    //

    (*arch_search).evolution_info.success_idx = max_success_idx + 1;

    //Set arrayfire random seed
    let mut tempseedarr = (*arch_search).evolution_info.crossval_vec.clone();
    tempseedarr.push(total_tries.to_f64().unwrap());

    set_network_seed2(&tempseedarr);

    // Search for network data using various methods
    let mut selected_idx = 0;

    match arch_search.evolution_info.search_strategy {
        evolution_search_type::TOP5_SEARCH => {
            selected_idx = search_top5(arch_search, &mut checked_paths);
        }
        evolution_search_type::METROPOLIS_SEARCH => {
            if max_success_idx <= 3 {
                selected_idx = search_top5(arch_search, &mut checked_paths);
            } else {
                selected_idx = search_metropolis(arch_search, &mut checked_paths, max_success_idx);
            }
        }
        evolution_search_type::SURROGATE_SPLINE_SEARCH => (),
    };

    //Generate new point
    let selected_netdata =
        clone_netdata(&((*arch_search).evolution_info.netdata_vec[selected_idx]));

    //Select active size
    let cur_active_size = selected_netdata.active_size.clone();

    let max_active_size = (*arch_search).evolution_info.max_active_size.clone();
    let min_active_size = (*arch_search).evolution_info.min_active_size.clone();

    let max_active_size_step = (*arch_search).evolution_info.max_active_size_step.clone();
    let min_active_size_step = (*arch_search).evolution_info.min_active_size_step.clone();

    let neuron_max = (cur_active_size as f64) * max_active_size_step;
    let neuron_min = (cur_active_size as f64) * min_active_size_step;

    let mut new_active_size = cur_active_size.clone();
    while 1 == 1 {
        let samplearr_dims = arrayfire::Dim4::new(&[100, 1, 1, 1]);
        let mut samplearr = arrayfire::randu::<f64>(samplearr_dims);
        samplearr = (samplearr) * (neuron_max - neuron_min) + neuron_min;

        let mut minusarr = arrayfire::randu::<f64>(samplearr_dims);
        minusarr = minusarr - 0.5f64;

        minusarr = 2.0f64 * (arrayfire::sign(&minusarr) - 0.5f64);

        samplearr = arrayfire::mul(&samplearr, &minusarr, false);

        samplearr = samplearr + (cur_active_size as f64);

        let mut samplearr_cpu = vec![f64::default(); samplearr.elements()];
        samplearr.host(&mut samplearr_cpu);

        for nsize in samplearr_cpu {
            if (nsize > 1.0f64) {
                let nsize_u64 = nsize as u64;
                if (min_active_size < nsize_u64) && (nsize_u64 < max_active_size) {
                    new_active_size = nsize_u64.clone();
                    break;
                }
            }
        }

        if (new_active_size != cur_active_size) {
            break;
        }
    }

    //Select proc_num
    let cur_proc_num = selected_netdata.proc_num.clone();

    let max_proc_num = (*arch_search).evolution_info.max_proc_num.clone();
    let min_proc_num = (*arch_search).evolution_info.min_proc_num.clone();

    let max_proc_num_step = (*arch_search).evolution_info.max_proc_num_step.clone();
    let min_proc_num_step = (*arch_search).evolution_info.min_proc_num_step.clone();

    let mut new_proc_num = cur_proc_num.clone();
    let mut breakflag = false;
    while 1 == 1 {
        let samplearr_dims = arrayfire::Dim4::new(&[100, 1, 1, 1]);
        let mut samplearr = arrayfire::randu::<f64>(samplearr_dims);
        samplearr = (samplearr) * (max_proc_num_step - min_proc_num_step) + min_proc_num_step;

        let mut minusarr = arrayfire::randu::<f64>(samplearr_dims);
        minusarr = minusarr - 0.5f64;

        minusarr = 2.0f64 * (arrayfire::sign(&minusarr) - 0.5f64);

        samplearr = arrayfire::mul(&samplearr, &minusarr, false);

        samplearr = samplearr + (cur_proc_num as f64);

        samplearr = arrayfire::round(&samplearr);

        let mut samplearr_cpu = vec![f64::default(); samplearr.elements()];
        samplearr.host(&mut samplearr_cpu);

        for pnum in samplearr_cpu {
            if (pnum > 1.0f64) {
                let pnum_u64 = pnum as u64;
                if (min_proc_num <= pnum_u64) && (pnum_u64 <= max_proc_num) {
                    new_proc_num = pnum_u64.clone();
                    breakflag = true;
                    break;
                }
            }
        }

        if (breakflag) {
            break;
        }
    }

    //Get filename
    let filename = checked_paths[selected_idx].clone();

    //Load network
    (*arch_search).neural_network = load_network2(&filename);

    //Load cube pos
    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let mut cube_glia_pos = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut cube_neuron_pos = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut cube_neuron_idx = arrayfire::constant::<i32>(0, temp_dims);

    load_network_structure(
        &cube_path,
        &mut cube_glia_pos,
        &mut cube_neuron_pos,
        &mut cube_neuron_idx,
    );
    drop(cube_glia_pos);
    drop(cube_neuron_idx);

    println!("Changes");
    println!("cur_active_size {}", cur_active_size);
    println!("cur_proc_num {}", cur_proc_num);

    println!("new_active_size {}", new_active_size);
    println!("new_proc_num {}", new_proc_num);

    //Change process num
    (*arch_search).neural_network.netdata.proc_num = new_proc_num;

    if (new_active_size > cur_active_size) {
        //Add new neurons
        add_neuron_to_existing2(&cube_neuron_pos, cube_radius, new_active_size, arch_search);
    } else if (new_active_size < cur_active_size) {
        //Make the neural network smaller
        reduce_network_size(new_active_size, arch_search);

        let rand_number = single_random_uniform();

        if rand_number > 0.5 {
            //Add new neurons
            add_neuron_to_existing2(
                &cube_neuron_pos,
                cube_radius,
                new_active_size + 2,
                arch_search,
            );
        }
    }

    //Save neural network
    let active_size = (*arch_search).neural_network.neuron_idx.dims()[0];
    let proc_num = (*arch_search).neural_network.netdata.proc_num.clone();
    let con_num = (*arch_search).neural_network.WColIdx.dims()[0];
    let dir_path = (*arch_search).evolution_info.dir_path.clone();
    (*arch_search).evolution_info.cur_path = format!(
        "{}/active_size_{}_proc_num_{}_con_num_{}.csv",
        dir_path, active_size, proc_num, con_num
    );

    save_network2(
        &((*arch_search).evolution_info.cur_path),
        &((*arch_search).neural_network),
    );
}
