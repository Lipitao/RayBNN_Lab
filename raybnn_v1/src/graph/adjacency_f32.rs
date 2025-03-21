extern crate arrayfire;
use crate::neural::network_f32::network_metadata_type;

use rand::distributions::{Distribution, Uniform};
use rayon::prelude::*;

use crate::graph::tree_u32::find_unique;
use nohash_hasher;
use rand::seq::SliceRandom;
use std::collections::HashMap;

use crate::graph::tree_i32::find_unique as find_unique_i32;

use super::large_sparse_i32::COO_batch_find;

const two: f32 = 2.0;
const COO_find_limit: u64 = 1500000000;

pub fn get_global_weight_idx(
    neuron_size: u64,
    WRowIdxCOO: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
) -> arrayfire::Array<u64> {
    (WRowIdxCOO.cast::<u64>() * (neuron_size)) + WColIdx.cast::<u64>()
}

pub fn get_global_weight_idx2(
    neuron_size: u64,
    WRowIdxCOO: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
) -> arrayfire::Array<u64> {
    WRowIdxCOO.cast::<u64>() + (WColIdx.cast::<u64>() * (neuron_size))
}

pub fn select_values(
    WValues: &mut arrayfire::Array<f32>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    sel: &arrayfire::Array<u32>,
) {
    *WValues = arrayfire::lookup(WValues, sel, 0);
    *WRowIdxCOO = arrayfire::lookup(WRowIdxCOO, sel, 0);
    *WColIdx = arrayfire::lookup(WColIdx, sel, 0);
}

pub fn clear_input(
    WValues: &mut arrayfire::Array<f32>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    input_rows: u64,
) {
    let single = input_rows as i32;

    //let cmp2 = (single <= WRowIdxCOO );
    let cmp1 = arrayfire::le(&single, WRowIdxCOO, false);

    let sel = arrayfire::locate(&cmp1);

    select_values(WValues, WRowIdxCOO, WColIdx, &sel);
}

pub fn clear_output(
    WValues: &mut arrayfire::Array<f32>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    output_cols: u64,
) {
    let single = output_cols as i32;

    //let cmp2 = (WColIdx < single );
    let cmp1 = arrayfire::lt(WColIdx, &single, false);

    let sel = arrayfire::locate(&cmp1);

    select_values(WValues, WRowIdxCOO, WColIdx, &sel);
}

pub fn clear_input_to_hidden(
    WValues: &mut arrayfire::Array<f32>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    input_cols: u64,
) {
    let single = input_cols as i32;

    //let cmp2 = (WColIdx >= single );
    let cmp1 = arrayfire::ge(WColIdx, &single, false);

    let sel = arrayfire::locate(&cmp1);

    select_values(WValues, WRowIdxCOO, WColIdx, &sel);
}

pub fn delete_smallest_weights(
    WValues: &mut arrayfire::Array<f32>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    del_num: u64,
) {
    let WValues_num = WValues.dims()[0];
    let abs = arrayfire::abs(&WValues);
    //Sort to find small weights
    let (_, mut idx) = arrayfire::sort_index(&abs, 0, false);

    //Select biggest weights
    let mut sel = arrayfire::rows(&idx, 0, (WValues_num - del_num - 1) as i64);

    sel = find_unique(&sel, WValues_num);

    //Select COO Matrix
    select_values(WValues, WRowIdxCOO, WColIdx, &sel);
}

pub fn add_random_weights(
    netdata: &network_metadata_type,

    neuron_idx: &arrayfire::Array<i32>,

    WValues: &mut arrayfire::Array<f32>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,

    add_num: u64,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();
    let batch_size: u64 = netdata.batch_size.clone();

    let mut abs = arrayfire::abs(&WValues);
    let (min_val, _) = arrayfire::min_all(&abs);

    //Compute global index
    let mut gidx1 = get_global_weight_idx(neuron_size, &WRowIdxCOO, &WColIdx);

    let mut gidx1_cpu = vec![u64::default(); gidx1.elements()];
    gidx1.host(&mut gidx1_cpu);

    let mut WValues_cpu = vec![f32::default(); WValues.elements()];
    WValues.host(&mut WValues_cpu);

    let mut WRowIdxCOO_cpu = vec![i32::default(); WRowIdxCOO.elements()];
    WRowIdxCOO.host(&mut WRowIdxCOO_cpu);

    let mut WColIdx_cpu = vec![i32::default(); WColIdx.elements()];
    WColIdx.host(&mut WColIdx_cpu);

    let neuron_dims = neuron_idx.dims();
    let neuron_num = neuron_dims[0];

    let hidden_idx = arrayfire::rows(
        neuron_idx,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );

    let input_idx = arrayfire::rows(&neuron_idx, 0, (input_size - 1) as i64);

    let output_idx = arrayfire::rows(
        &neuron_idx,
        (neuron_idx.dims()[0] - output_size) as i64,
        (neuron_idx.dims()[0] - 1) as i64,
    );

    let mut hidden_idx_cpu = vec![i32::default(); hidden_idx.elements()];
    hidden_idx.host(&mut hidden_idx_cpu);

    let mut input_idx_cpu = vec![i32::default(); input_idx.elements()];
    input_idx.host(&mut input_idx_cpu);

    let mut output_idx_cpu = vec![i32::default(); output_idx.elements()];
    output_idx.host(&mut output_idx_cpu);

    let mut join_WValues = nohash_hasher::IntMap::default();
    let mut join_WColIdx = nohash_hasher::IntMap::default();
    let mut join_WRowIdxCOO = nohash_hasher::IntMap::default();

    for qq in 0..gidx1.elements() {
        let cur_gidx = gidx1_cpu[qq].clone();

        join_WValues.insert(cur_gidx, WValues_cpu[qq].clone());
        join_WColIdx.insert(cur_gidx, WColIdx_cpu[qq].clone());
        join_WRowIdxCOO.insert(cur_gidx, WRowIdxCOO_cpu[qq].clone());
    }

    let mut rng = rand::thread_rng();
    let choose_connection = Uniform::from(0.0..1.0f32);
    let value_range = Uniform::from(-min_val..min_val);

    let p1 = (input_size as f32) / (neuron_num as f32);
    let p2 = ((input_size + hidden_idx.dims()[0]) as f32) / (neuron_num as f32);

    let mut add_counter = 0;
    while 1 == 1 {
        let connection_type = choose_connection.sample(&mut rng);

        let mut cur_rows = 0;
        let mut cur_cols = 0;

        if connection_type <= p1 {
            //Input to Hidden
            cur_rows = hidden_idx_cpu
                .choose(&mut rand::thread_rng())
                .unwrap()
                .clone();
            cur_cols = input_idx_cpu
                .choose(&mut rand::thread_rng())
                .unwrap()
                .clone();
        } else if (p1 < connection_type) && (connection_type <= p2) {
            //Hidden to Hidden
            cur_rows = hidden_idx_cpu
                .choose(&mut rand::thread_rng())
                .unwrap()
                .clone();
            cur_cols = hidden_idx_cpu
                .choose(&mut rand::thread_rng())
                .unwrap()
                .clone();
        } else {
            //Hidden to Output
            cur_rows = output_idx_cpu
                .choose(&mut rand::thread_rng())
                .unwrap()
                .clone();
            cur_cols = hidden_idx_cpu
                .choose(&mut rand::thread_rng())
                .unwrap()
                .clone();
        }

        let cur_gidx = ((cur_rows as u64) * (neuron_size)) + (cur_cols as u64);
        if join_WValues.contains_key(&cur_gidx) == false {
            let new_value = value_range.sample(&mut rng);
            join_WValues.insert(cur_gidx, new_value);
            join_WColIdx.insert(cur_gidx, cur_cols.clone());
            join_WRowIdxCOO.insert(cur_gidx, cur_rows.clone());

            add_counter = add_counter + 1;

            if add_counter >= add_num {
                break;
            }
        }
    }

    let mut gidx3: Vec<u64> = join_WValues.clone().into_keys().collect();
    gidx3.par_sort_unstable();

    WValues_cpu = Vec::new();
    WRowIdxCOO_cpu = Vec::new();
    WColIdx_cpu = Vec::new();

    for qq in gidx3 {
        WValues_cpu.push(join_WValues[&qq].clone());
        WColIdx_cpu.push(join_WColIdx[&qq].clone());
        WRowIdxCOO_cpu.push(join_WRowIdxCOO[&qq].clone());
    }

    *WValues = arrayfire::Array::new(
        &WValues_cpu,
        arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]),
    );
    *WColIdx = arrayfire::Array::new(
        &WColIdx_cpu,
        arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]),
    );
    *WRowIdxCOO = arrayfire::Array::new(
        &WRowIdxCOO_cpu,
        arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]),
    );
}

pub fn delete_weights_with_prob(
    WValues: &mut arrayfire::Array<f32>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    del_num: u64,
) {
    let WValues_num = WValues.dims()[0];
    let mut abs = arrayfire::abs(&WValues);

    let randarr = arrayfire::randu::<f32>(abs.dims());

    abs = arrayfire::mul(&abs, &randarr, false);

    //Sort to find small weights
    let (_, mut idx) = arrayfire::sort_index(&abs, 0, false);

    //Select biggest weights
    let mut sel = arrayfire::rows(&idx, 0, (WValues_num - del_num - 1) as i64);

    sel = find_unique(&sel, WValues_num);

    //Select COO Matrix
    select_values(WValues, WRowIdxCOO, WColIdx, &sel);
}

pub fn select_forward_sphere(
    netdata: &network_metadata_type,
    WValues: &mut arrayfire::Array<f32>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    neuron_pos: &arrayfire::Array<f32>,
    neuron_idx: &arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();

    let colseq = arrayfire::Seq::new(0.0, (space_dims - 1) as f32, 1.0);

    let mut temparr =
        arrayfire::constant::<f32>(0.0, arrayfire::Dim4::new(&[neuron_size, space_dims, 1, 1]));

    let idx = neuron_idx.clone();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idx, 0, None);
    idxrs.set_index(&colseq, 1, Some(false));
    arrayfire::assign_gen(&mut temparr, &idxrs, neuron_pos);

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(WRowIdxCOO, 0, None);
    idxrs.set_index(&colseq, 1, Some(false));
    let row_neuron_pos = arrayfire::index_gen(&temparr, idxrs);

    let mut row_magsq = arrayfire::pow(&row_neuron_pos, &two, false);
    row_magsq = arrayfire::sum(&row_magsq, 1);

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(WColIdx, 0, None);
    idxrs.set_index(&colseq, 1, Some(false));
    let col_neuron_pos = arrayfire::index_gen(&temparr, idxrs);

    let mut col_magsq = arrayfire::pow(&col_neuron_pos, &two, false);
    col_magsq = arrayfire::sum(&col_magsq, 1);

    //let cmp1 = (WRowIdxCOO < WColIdx);
    let cmp1 = arrayfire::lt(&row_magsq, &col_magsq, false);

    let sel = arrayfire::locate(&cmp1);

    select_values(WValues, WRowIdxCOO, WColIdx, &sel);
}

pub fn delete_neurons_at_idx(
    delete_idx: &arrayfire::Array<i32>,

    WValues: &mut arrayfire::Array<f32>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let COO_batch_size = 1 + ((COO_find_limit / WColIdx.dims()[0]) as u64);
    let valsel = COO_batch_find(WColIdx, &delete_idx, COO_batch_size);

    let mut temparr = arrayfire::constant::<bool>(true, WColIdx.dims());

    let ones = arrayfire::constant::<bool>(false, valsel.dims());

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&valsel, 0, None);
    arrayfire::assign_gen(&mut temparr, &idxrs, &ones);

    let valsel2 = arrayfire::locate(&temparr);

    select_values(WValues, WRowIdxCOO, WColIdx, &valsel2);
}

pub fn delete_unused_neurons(
    netdata: &network_metadata_type,
    WValues: &mut arrayfire::Array<f32>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    glia_pos: &mut arrayfire::Array<f32>,
    neuron_pos: &mut arrayfire::Array<f32>,
    neuron_idx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();

    //Get active non zero cols
    let mut temparr =
        arrayfire::constant::<bool>(false, arrayfire::Dim4::new(&[neuron_size, 1, 1, 1]));

    let ones = arrayfire::constant::<bool>(true, WColIdx.dims());

    let idx = WColIdx.clone();
    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idx, 0, None);
    arrayfire::assign_gen(&mut temparr, &idxrs, &ones);

    //Get all non zero index of col
    let mut sel = arrayfire::locate(&temparr).cast::<i32>();

    let active_size = neuron_idx.dims()[0];
    let output_idx = arrayfire::rows(
        neuron_idx,
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );
    //Add output neurons to index
    sel = arrayfire::join(0, &sel, &output_idx);

    sel = find_unique_i32(&sel, neuron_size);

    let update_neuron_idx = sel.clone();

    let COO_batch_size = 1 + ((COO_find_limit / WRowIdxCOO.dims()[0]) as u64);

    let valsel = COO_batch_find(WRowIdxCOO, &sel, COO_batch_size).cast::<u32>();

    if (valsel.dims()[0] == WRowIdxCOO.dims()[0]) {
        return;
    }

    select_values(WValues, WRowIdxCOO, WColIdx, &valsel);

    let mut temparr =
        arrayfire::constant::<f32>(0.0, arrayfire::Dim4::new(&[neuron_size, space_dims, 1, 1]));

    let seq1 = arrayfire::Seq::new(0.0, (space_dims - 1) as f32, 1.0);
    let idx = neuron_idx.clone();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idx, 0, None);
    idxrs.set_index(&seq1, 1, Some(false));
    arrayfire::assign_gen(&mut temparr, &idxrs, neuron_pos);

    let mut temparr2 =
        arrayfire::constant::<bool>(false, arrayfire::Dim4::new(&[neuron_size, 1, 1, 1]));
    let ones = arrayfire::constant::<bool>(true, idx.dims());

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idx, 0, None);
    arrayfire::assign_gen(&mut temparr2, &idxrs, &ones);

    let zeros = arrayfire::constant::<bool>(false, sel.dims());
    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&sel, 0, None);
    arrayfire::assign_gen(&mut temparr2, &idxrs, &zeros);

    let new_glia_idx = arrayfire::locate(&temparr2);
    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&new_glia_idx, 0, None);
    idxrs.set_index(&seq1, 1, Some(false));
    let new_glia_pos = arrayfire::index_gen(&temparr, idxrs);

    *glia_pos = arrayfire::join(0, glia_pos, &new_glia_pos);

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&sel, 0, None);
    idxrs.set_index(&seq1, 1, Some(false));
    *neuron_pos = arrayfire::index_gen(&temparr, idxrs);

    *neuron_idx = update_neuron_idx;
}

pub fn delete_smallest_neurons(
    netdata: &network_metadata_type,
    neuron_idx: &arrayfire::Array<i32>,
    del_num: u64,

    WValues: &mut arrayfire::Array<f32>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();

    let in_idx = arrayfire::rows(neuron_idx, 0, (input_size - 1) as i64);

    let out_idx = arrayfire::rows(
        neuron_idx,
        (neuron_idx.dims()[0] - output_size) as i64,
        (neuron_idx.dims()[0] - 1) as i64,
    );

    let (max_in_idx, _) = arrayfire::max_all(&in_idx);

    let (min_out_idx, _) = arrayfire::min_all(&out_idx);

    //  (WColIdx > max_in_idx )
    let mut CMPRET = arrayfire::gt(WColIdx, &max_in_idx, false);

    //  (min_out_idx > WColIdx )
    let CMP1 = arrayfire::gt(&min_out_idx, WColIdx, false);
    CMPRET = arrayfire::and(&CMPRET, &CMP1, false);

    let selidx = arrayfire::locate(&CMPRET);

    let newWColIdx = arrayfire::lookup(WColIdx, &selidx, 0);

    let newWValues = arrayfire::lookup(WValues, &selidx, 0);

    //let WValues_num  = WValues.dims()[0];
    let abs = arrayfire::abs(&newWValues);

    let (keys, values) = arrayfire::sum_by_key(&newWColIdx, &abs, 0);

    //Sort to find small neurons
    let (_, mut idx) = arrayfire::sort_index(&values, 0, false);

    //Select biggest neurons
    let idxnum = idx.dims()[0];
    let mut sel = arrayfire::rows(&idx, 0, (idxnum - del_num - 1) as i64);

    let mut newkeys = arrayfire::lookup(&keys, &sel, 0);

    newkeys = arrayfire::join(0, &in_idx, &newkeys);

    newkeys = find_unique_i32(&newkeys, neuron_size);

    let COO_batch_size = 1 + ((COO_find_limit / WColIdx.dims()[0]) as u64);

    let valsel = COO_batch_find(WColIdx, &newkeys, COO_batch_size).cast::<u32>();

    if (valsel.dims()[0] == 0) {
        return;
    }

    if (valsel.dims()[0] == WColIdx.dims()[0]) {
        return;
    }

    //Select COO Matrix
    select_values(WValues, WRowIdxCOO, WColIdx, &valsel);
}

pub fn delete_smallest_neurons_with_prob(
    netdata: &network_metadata_type,
    neuron_idx: &arrayfire::Array<i32>,
    del_num: u64,

    WValues: &mut arrayfire::Array<f32>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();

    let in_idx = arrayfire::rows(neuron_idx, 0, (input_size - 1) as i64);

    let out_idx = arrayfire::rows(
        neuron_idx,
        (neuron_idx.dims()[0] - output_size) as i64,
        (neuron_idx.dims()[0] - 1) as i64,
    );

    let (max_in_idx, _) = arrayfire::max_all(&in_idx);

    let (min_out_idx, _) = arrayfire::min_all(&out_idx);

    //  (WColIdx > max_in_idx )
    let mut CMPRET = arrayfire::gt(WColIdx, &max_in_idx, false);

    //  (min_out_idx > WColIdx )
    let CMP1 = arrayfire::gt(&min_out_idx, WColIdx, false);
    CMPRET = arrayfire::and(&CMPRET, &CMP1, false);

    let selidx = arrayfire::locate(&CMPRET);

    let newWColIdx = arrayfire::lookup(WColIdx, &selidx, 0);

    let newWValues = arrayfire::lookup(WValues, &selidx, 0);

    //let WValues_num  = WValues.dims()[0];
    let abs = arrayfire::abs(&newWValues);

    let (keys, mut values) = arrayfire::sum_by_key(&newWColIdx, &abs, 0);

    let randarr = arrayfire::randu::<f32>(values.dims());

    values = arrayfire::mul(&values, &randarr, false);

    //Sort to find small neurons
    let (_, mut idx) = arrayfire::sort_index(&values, 0, false);

    //Select biggest neurons
    let idxnum = idx.dims()[0];
    let mut sel = arrayfire::rows(&idx, 0, (idxnum - del_num - 1) as i64);

    let mut newkeys = arrayfire::lookup(&keys, &sel, 0);

    newkeys = arrayfire::join(0, &in_idx, &newkeys);

    newkeys = find_unique_i32(&newkeys, neuron_size);

    let COO_batch_size = 1 + ((COO_find_limit / WColIdx.dims()[0]) as u64);

    let valsel = COO_batch_find(WColIdx, &newkeys, COO_batch_size).cast::<u32>();

    if (valsel.dims()[0] == 0) {
        return;
    }

    if (valsel.dims()[0] == WColIdx.dims()[0]) {
        return;
    }

    //Select COO Matrix
    select_values(WValues, WRowIdxCOO, WColIdx, &valsel);
}

pub fn select_neurons(
    sel_idx: &arrayfire::Array<i32>,
    neuron_size: u64,
    neuron_pos: &mut arrayfire::Array<f32>,
    neuron_idx: &mut arrayfire::Array<i32>,
) {
    let space_dims = neuron_pos.dims()[1];
    let mut temparr =
        arrayfire::constant::<f32>(0.0, arrayfire::Dim4::new(&[neuron_size, space_dims, 1, 1]));

    let seq1 = arrayfire::Seq::new(0.0, (space_dims - 1) as f32, 1.0);
    let in_idx = neuron_idx.clone();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&in_idx, 0, None);
    idxrs.set_index(&seq1, 1, Some(false));
    arrayfire::assign_gen(&mut temparr, &idxrs, neuron_pos);

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(sel_idx, 0, None);
    idxrs.set_index(&seq1, 1, Some(false));

    *neuron_pos = arrayfire::index_gen(&temparr, idxrs);
    *neuron_idx = sel_idx.clone();
}
