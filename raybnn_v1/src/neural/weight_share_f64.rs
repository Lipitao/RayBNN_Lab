extern crate arrayfire;

use nohash_hasher;
use std::collections::HashMap;

use crate::graph::large_sparse_i32::COO_batch_find;

use crate::graph::adjacency_f64::select_values;

const zero: f64 = 0.0;

const COO_find_limit: u64 = 1500000000;

pub fn limit_cols(
    dist_UAF_start_map: &nohash_hasher::IntMap<u64, i64>,
    dist_UAF_end_map: &nohash_hasher::IntMap<u64, i64>,
    dist_UAF_num_map: &nohash_hasher::IntMap<u64, u64>,
    total_num: u64,
    max_cols: u64,

    WValues: &mut arrayfire::Array<f64>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let COO_batch_size = 1 + ((COO_find_limit / WRowIdxCOO.dims()[0]) as u64);

    let max_cols_i32 = max_cols as i32;
    let max_cols_usize = max_cols as usize;

    let mut dist_start = dist_UAF_start_map[&0];
    let mut dist_end = dist_UAF_end_map[&0];
    let mut dist_num = dist_UAF_num_map[&0];

    let mut block_size = dist_end - dist_start + 1;

    let mut N_dims = arrayfire::Dim4::new(&[dist_num * (block_size as u64), 1, 1, 1]);
    let repeat_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut keys = (dist_start as i32) + arrayfire::iota::<i32>(N_dims, repeat_dims);

    for i in 1..total_num {
        dist_start = dist_UAF_start_map[&i];
        dist_end = dist_UAF_end_map[&i];
        dist_num = dist_UAF_num_map[&i];

        block_size = dist_end - dist_start + 1;

        N_dims = arrayfire::Dim4::new(&[dist_num * (block_size as u64), 1, 1, 1]);

        let newkeys = (dist_start as i32) + arrayfire::iota::<i32>(N_dims, repeat_dims);

        keys = arrayfire::join(0, &keys, &newkeys);
    }

    let mut temparr = arrayfire::constant::<bool>(true, WValues.dims());

    let keys_size = keys.dims()[0] as i64;

    for i in 0..keys_size {
        let cur_key = arrayfire::row(&keys, i);

        //Get indexes of WValues
        let mut valsel = COO_batch_find(WRowIdxCOO, &cur_key, COO_batch_size);

        if (valsel.dims()[0] <= max_cols) {
            continue;
        }

        let mut valsel_cpu = vec![i32::default(); valsel.elements()];

        valsel.host(&mut valsel_cpu);

        let valsel_len = valsel_cpu.len();

        let ones = arrayfire::constant::<bool>(
            false,
            arrayfire::Dim4::new(&[(valsel_len as u64) - max_cols, 1, 1, 1]),
        );

        let first = valsel_cpu[max_cols_usize] as u32;
        let last = valsel_cpu[valsel_len - 1] as u32;

        let seqs = &[arrayfire::Seq::new(first, last, 1 as u32)];
        arrayfire::assign_seq(&mut temparr, seqs, &ones);
    }

    let idx2 = arrayfire::locate(&temparr);

    select_values(WValues, WRowIdxCOO, WColIdx, &idx2);
}

pub fn pad_cols(
    dist_UAF_start_map: &nohash_hasher::IntMap<u64, i64>,
    dist_UAF_end_map: &nohash_hasher::IntMap<u64, i64>,
    dist_UAF_num_map: &nohash_hasher::IntMap<u64, u64>,
    total_num: u64,
    input_size: u64,
    max_cols: u64,

    WValues: &mut arrayfire::Array<f64>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let COO_batch_size = 1 + ((COO_find_limit / WRowIdxCOO.dims()[0]) as u64);

    let max_cols_i32 = max_cols as i32;
    let max_cols_usize = max_cols as usize;

    let mut dist_start = dist_UAF_start_map[&0];
    let mut dist_end = dist_UAF_end_map[&0];
    let mut dist_num = dist_UAF_num_map[&0];

    let mut block_size = dist_end - dist_start + 1;

    let mut N_dims = arrayfire::Dim4::new(&[dist_num * (block_size as u64), 1, 1, 1]);
    let repeat_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut keys = (dist_start as i32) + arrayfire::iota::<i32>(N_dims, repeat_dims);

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut delete_first = true;
    let mut newWValues = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut newWRowIdxCOO = arrayfire::constant::<i32>(0, temp_dims);
    let mut newWColIdx = arrayfire::constant::<i32>(0, temp_dims);

    let cur_key = arrayfire::row(&keys, 0);

    //Get indexes of WValues
    let mut valsel = COO_batch_find(WRowIdxCOO, &cur_key, COO_batch_size);

    if (valsel.dims()[0] > 0) {
        let mut valsel_cpu = vec![i32::default(); valsel.elements()];
        valsel.host(&mut valsel_cpu);

        let temp_start = valsel_cpu[0] as i64;

        if (temp_start > 0) {
            delete_first = false;
            newWValues = arrayfire::rows(WValues, 0, temp_start - 1);
            newWRowIdxCOO = arrayfire::rows(WRowIdxCOO, 0, temp_start - 1);
            newWColIdx = arrayfire::rows(WColIdx, 0, temp_start - 1);
        }
    }

    for i in 1..total_num {
        dist_start = dist_UAF_start_map[&i];
        dist_end = dist_UAF_end_map[&i];
        dist_num = dist_UAF_num_map[&i];

        block_size = dist_end - dist_start + 1;

        N_dims = arrayfire::Dim4::new(&[dist_num * (block_size as u64), 1, 1, 1]);

        let newkeys = (dist_start as i32) + arrayfire::iota::<i32>(N_dims, repeat_dims);

        keys = arrayfire::join(0, &keys, &newkeys);
    }

    let keys_size = keys.dims()[0] as i64;

    let mut last = 0;

    for i in 0..keys_size {
        let cur_key = arrayfire::row(&keys, i);

        //Get indexes of WValues
        let mut valsel = COO_batch_find(WRowIdxCOO, &cur_key, COO_batch_size);

        if (valsel.dims()[0] == 0) {
            let values_dims = arrayfire::Dim4::new(&[max_cols, 1, 1, 1]);
            let insert_values = arrayfire::constant::<f64>(zero, values_dims);

            let insert_rows = arrayfire::tile(&cur_key, values_dims);

            let cols_dims = arrayfire::Dim4::new(&[input_size, 1, 1, 1]);

            let cols_rand = arrayfire::randu::<f64>(cols_dims);

            let (_, idx) = arrayfire::sort_index(&cols_rand, 0, false);

            let insert_cols = arrayfire::rows(&idx, 0, (max_cols - 1) as i64).cast::<i32>();

            newWValues = arrayfire::join(0, &newWValues, &insert_values);
            newWRowIdxCOO = arrayfire::join(0, &newWRowIdxCOO, &insert_rows);
            newWColIdx = arrayfire::join(0, &newWColIdx, &insert_cols);

            continue;
        }

        if (valsel.dims()[0] >= max_cols) {
            let insert_values = arrayfire::lookup(WValues, &valsel, 0);

            let insert_rows = arrayfire::lookup(WRowIdxCOO, &valsel, 0);

            let insert_cols = arrayfire::lookup(WColIdx, &valsel, 0);

            newWValues = arrayfire::join(0, &newWValues, &insert_values);
            newWRowIdxCOO = arrayfire::join(0, &newWRowIdxCOO, &insert_rows);
            newWColIdx = arrayfire::join(0, &newWColIdx, &insert_cols);

            let mut valsel_cpu = vec![i32::default(); valsel.elements()];
            valsel.host(&mut valsel_cpu);

            let valsel_len = valsel_cpu.len();

            last = valsel_cpu[valsel_len - 1] as i64;

            continue;
        }

        //Get rows of WRowIdx
        let rvec = arrayfire::lookup(WRowIdxCOO, &valsel, 0);

        let cvec = arrayfire::lookup(WColIdx, &valsel, 0);

        let mut rvec_cpu = vec![i32::default(); rvec.elements()];
        rvec.host(&mut rvec_cpu);

        let mut valsel_cpu = vec![i32::default(); valsel.elements()];
        valsel.host(&mut valsel_cpu);

        let valsel_len = valsel_cpu.len();

        let first = valsel_cpu[0] as i64;
        last = valsel_cpu[valsel_len - 1] as i64;

        let values_dims = arrayfire::Dim4::new(&[max_cols - (valsel_len as u64), 1, 1, 1]);

        let mut insert_values = arrayfire::rows(WValues, first, last);

        let new_values = arrayfire::constant::<f64>(zero, values_dims);

        insert_values = arrayfire::join(0, &insert_values, &new_values);

        let rows_dims = arrayfire::Dim4::new(&[max_cols, 1, 1, 1]);

        let insert_rows = arrayfire::constant::<i32>(rvec_cpu[0], rows_dims);

        let cols_dims = arrayfire::Dim4::new(&[input_size, 1, 1, 1]);

        let mut cols_rand = arrayfire::randu::<f64>(cols_dims);

        let inarr = arrayfire::constant::<f64>(zero, cvec.dims());

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&cvec, 0, None);
        arrayfire::assign_gen(&mut cols_rand, &idxrs, &inarr);

        let (_, idx) = arrayfire::sort_index(&cols_rand, 0, false);

        let mut insert_cols = arrayfire::rows(WColIdx, first, last);

        let new_cols =
            arrayfire::rows(&idx, 0, (max_cols as i64) - (valsel_len as i64) - 1).cast::<i32>();

        insert_cols = arrayfire::join(0, &insert_cols, &new_cols);

        let (_, idx2) = arrayfire::sort_index(&insert_cols, 0, true);

        insert_cols = arrayfire::lookup(&insert_cols, &idx2, 0);

        insert_values = arrayfire::lookup(&insert_values, &idx2, 0);

        newWValues = arrayfire::join(0, &newWValues, &insert_values);
        newWRowIdxCOO = arrayfire::join(0, &newWRowIdxCOO, &insert_rows);
        newWColIdx = arrayfire::join(0, &newWColIdx, &insert_cols);
    }

    if delete_first == true {
        let newWValues_len = newWValues.dims()[0] as i64;

        newWValues = arrayfire::rows(&newWValues, 1, newWValues_len - 1);
        newWRowIdxCOO = arrayfire::rows(&newWRowIdxCOO, 1, newWValues_len - 1);
        newWColIdx = arrayfire::rows(&newWColIdx, 1, newWValues_len - 1);
    }

    let WValues_len = WValues.dims()[0] as i64;

    if (WValues_len - 1) == last {
        *WValues = newWValues;
        *WRowIdxCOO = newWRowIdxCOO;
        *WColIdx = newWColIdx;
    } else {
        *WValues = arrayfire::rows(WValues, last + 1, WValues_len - 1);
        *WRowIdxCOO = arrayfire::rows(WRowIdxCOO, last + 1, WValues_len - 1);
        *WColIdx = arrayfire::rows(WColIdx, last + 1, WValues_len - 1);

        *WValues = arrayfire::join(0, &newWValues, WValues);
        *WRowIdxCOO = arrayfire::join(0, &newWRowIdxCOO, WRowIdxCOO);
        *WColIdx = arrayfire::join(0, &newWColIdx, WColIdx);
    }
}

pub fn setup_distribute(
    dist_start_map: &nohash_hasher::IntMap<u64, i64>,
    dist_end_map: &nohash_hasher::IntMap<u64, i64>,
    dist_num_map: &nohash_hasher::IntMap<u64, u64>,
    total_num: u64,
    max_cols: u64,

    WValues: &mut arrayfire::Array<f64>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let mut dist_start = dist_start_map[&0];
    let mut dist_end = dist_end_map[&0];
    let mut dist_num = dist_num_map[&0];

    for i in 0..total_num {
        dist_start = dist_start_map[&i];
        dist_end = dist_end_map[&i];
        dist_num = dist_num_map[&i];

        let mut new_values = arrayfire::rows(WValues, dist_start, dist_end);
        let mut new_rows = arrayfire::rows(WRowIdxCOO, dist_start, dist_end);
        let mut new_cols = arrayfire::rows(WColIdx, dist_start, dist_end);

        let tile_dims = arrayfire::Dim4::new(&[dist_num, 1, 1, 1]);
        new_values = arrayfire::tile(&new_values, tile_dims);

        let row_num = new_rows.dims()[0] / max_cols;

        let repeat_dims = arrayfire::Dim4::new(&[new_rows.dims()[0], 1, 1, 1]);
        let tile_dims3 = arrayfire::Dim4::new(&[1, dist_num, 1, 1]);

        let count = (row_num as i32) * arrayfire::iota::<i32>(tile_dims3, repeat_dims);

        new_rows = arrayfire::tile(&new_rows, tile_dims) + arrayfire::flat(&count);

        let mut new_cols2 = new_cols.clone();

        for j in 1..dist_num {
            new_cols2 = arrayfire::shift(&new_cols2, &[max_cols as i32, 1, 1, 1]);
            new_cols = arrayfire::join(0, &new_cols, &new_cols2);
        }

        let block_size = dist_end - dist_start + 1;
        let last_idx = dist_start + ((dist_num as i64) * block_size) - 1;

        let seqs = &[arrayfire::Seq::new(
            dist_start as f64,
            last_idx as f64,
            1.0 as f64,
        )];
        arrayfire::assign_seq(WValues, seqs, &new_values);

        let seqs = &[arrayfire::Seq::new(
            dist_start as i32,
            last_idx as i32,
            1 as i32,
        )];
        arrayfire::assign_seq(WRowIdxCOO, seqs, &new_rows);
        arrayfire::assign_seq(WColIdx, seqs, &new_cols);
    }
}

pub fn setup_distribute_UAF(
    dist_UAF_start_map: &nohash_hasher::IntMap<u64, i64>,
    dist_UAF_end_map: &nohash_hasher::IntMap<u64, i64>,
    dist_UAF_num_map: &nohash_hasher::IntMap<u64, u64>,
    total_num: u64,

    H: &mut arrayfire::Array<f64>,
    A: &mut arrayfire::Array<f64>,
    B: &mut arrayfire::Array<f64>,
    C: &mut arrayfire::Array<f64>,
    D: &mut arrayfire::Array<f64>,
    E: &mut arrayfire::Array<f64>,
) {
    let mut dist_start = dist_UAF_start_map[&0];
    let mut dist_end = dist_UAF_end_map[&0];
    let mut dist_num = dist_UAF_num_map[&0];

    for i in 0..total_num {
        dist_start = dist_UAF_start_map[&i];
        dist_end = dist_UAF_end_map[&i];
        dist_num = dist_UAF_num_map[&i];

        let mut copy_H = arrayfire::rows(H, dist_start, dist_end);
        let mut copy_A = arrayfire::rows(A, dist_start, dist_end);
        let mut copy_B = arrayfire::rows(B, dist_start, dist_end);
        let mut copy_C = arrayfire::rows(C, dist_start, dist_end);
        let mut copy_D = arrayfire::rows(D, dist_start, dist_end);
        let mut copy_E = arrayfire::rows(E, dist_start, dist_end);

        let tile_dims = arrayfire::Dim4::new(&[dist_num, 1, 1, 1]);
        copy_H = arrayfire::tile(&copy_H, tile_dims);
        copy_A = arrayfire::tile(&copy_A, tile_dims);
        copy_B = arrayfire::tile(&copy_B, tile_dims);
        copy_C = arrayfire::tile(&copy_C, tile_dims);
        copy_D = arrayfire::tile(&copy_D, tile_dims);
        copy_E = arrayfire::tile(&copy_E, tile_dims);

        let block_size = dist_end - dist_start + 1;
        let last_idx = dist_start + ((dist_num as i64) * block_size) - 1;

        let seqs = &[arrayfire::Seq::new(
            dist_start as f64,
            last_idx as f64,
            1.0 as f64,
        )];
        arrayfire::assign_seq(H, seqs, &copy_H);
        arrayfire::assign_seq(A, seqs, &copy_A);
        arrayfire::assign_seq(B, seqs, &copy_B);
        arrayfire::assign_seq(C, seqs, &copy_C);
        arrayfire::assign_seq(D, seqs, &copy_D);
        arrayfire::assign_seq(E, seqs, &copy_E);
    }
}

pub fn sum_and_distribute(
    dist_start_map: &nohash_hasher::IntMap<u64, i64>,
    dist_end_map: &nohash_hasher::IntMap<u64, i64>,
    dist_num_map: &nohash_hasher::IntMap<u64, u64>,
    total_num: u64,

    alpha0: f64,
    gW: &arrayfire::Array<f64>,

    WValues: &mut arrayfire::Array<f64>,
) {
    let mut dist_start = dist_start_map[&0];
    let mut dist_end = dist_end_map[&0];
    let mut dist_num = dist_num_map[&0];

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut newWValues = arrayfire::constant::<f64>(0.0, temp_dims);

    if (dist_start > 0) {
        newWValues = arrayfire::rows(WValues, 0, dist_start - 1)
            + (alpha0 * -arrayfire::rows(gW, 0, dist_start - 1));

        let seqs = &[arrayfire::Seq::new(
            0.0 as f64,
            (dist_start - 1) as f64,
            1.0 as f64,
        )];
        arrayfire::assign_seq(WValues, seqs, &newWValues);
    }

    let mut last_idx = 0;

    for i in 0..total_num {
        dist_start = dist_start_map[&i];
        dist_end = dist_end_map[&i];
        dist_num = dist_num_map[&i];

        let block_size = dist_end - dist_start + 1;
        last_idx = dist_start + ((dist_num as i64) * block_size) - 1;

        let mut new_gW = arrayfire::rows(gW, dist_start, last_idx);

        let new_dims = arrayfire::Dim4::new(&[block_size as u64, dist_num, 1, 1]);
        new_gW = arrayfire::moddims(&new_gW, new_dims);

        new_gW = arrayfire::mean(&new_gW, 1);

        let mut new_values = arrayfire::rows(WValues, dist_start, dist_end) + (alpha0 * -new_gW);

        let tile_dims = arrayfire::Dim4::new(&[dist_num, 1, 1, 1]);
        new_values = arrayfire::tile(&new_values, tile_dims);

        let seqs = &[arrayfire::Seq::new(
            dist_start as f64,
            last_idx as f64,
            1.0 as f64,
        )];
        arrayfire::assign_seq(WValues, seqs, &new_values);
    }

    let WValues_len = WValues.dims()[0] as i64;

    newWValues = arrayfire::rows(WValues, last_idx + 1, WValues_len - 1)
        + (alpha0 * -arrayfire::rows(gW, last_idx + 1, WValues_len - 1));

    let seqs = &[arrayfire::Seq::new(
        (last_idx + 1) as f64,
        (WValues_len - 1) as f64,
        1.0 as f64,
    )];
    arrayfire::assign_seq(WValues, seqs, &newWValues);
}

pub fn sum_and_distribute_UAF(
    dist_UAF_start_map: &nohash_hasher::IntMap<u64, i64>,
    dist_UAF_end_map: &nohash_hasher::IntMap<u64, i64>,
    dist_UAF_num_map: &nohash_hasher::IntMap<u64, u64>,
    total_num: u64,

    alpha0: f64,
    alpha1: f64,
    gH: &arrayfire::Array<f64>,
    gA: &arrayfire::Array<f64>,
    gB: &arrayfire::Array<f64>,
    gC: &arrayfire::Array<f64>,
    gD: &arrayfire::Array<f64>,
    gE: &arrayfire::Array<f64>,

    H: &mut arrayfire::Array<f64>,
    A: &mut arrayfire::Array<f64>,
    B: &mut arrayfire::Array<f64>,
    C: &mut arrayfire::Array<f64>,
    D: &mut arrayfire::Array<f64>,
    E: &mut arrayfire::Array<f64>,
) {
    let mut dist_start = dist_UAF_start_map[&0];
    let mut dist_end = dist_UAF_end_map[&0];
    let mut dist_num = dist_UAF_num_map[&0];

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut newH = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut newA = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut newB = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut newC = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut newD = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut newE = arrayfire::constant::<f64>(0.0, temp_dims);

    if (dist_start > 0) {
        newH = arrayfire::rows(H, 0, dist_start - 1)
            + (alpha0 * -arrayfire::rows(gH, 0, dist_start - 1));
        newA = arrayfire::rows(A, 0, dist_start - 1)
            + (alpha1 * -arrayfire::rows(gA, 0, dist_start - 1));
        newB = arrayfire::rows(B, 0, dist_start - 1)
            + (alpha1 * -arrayfire::rows(gB, 0, dist_start - 1));
        newC = arrayfire::rows(C, 0, dist_start - 1)
            + (alpha1 * -arrayfire::rows(gC, 0, dist_start - 1));
        newD = arrayfire::rows(D, 0, dist_start - 1)
            + (alpha1 * -arrayfire::rows(gD, 0, dist_start - 1));
        newE = arrayfire::rows(E, 0, dist_start - 1)
            + (alpha1 * -arrayfire::rows(gE, 0, dist_start - 1));

        let seqs = &[arrayfire::Seq::new(
            0.0 as f64,
            (dist_start - 1) as f64,
            1.0 as f64,
        )];
        arrayfire::assign_seq(H, seqs, &newH);
        arrayfire::assign_seq(A, seqs, &newA);
        arrayfire::assign_seq(B, seqs, &newB);
        arrayfire::assign_seq(C, seqs, &newC);
        arrayfire::assign_seq(D, seqs, &newD);
        arrayfire::assign_seq(E, seqs, &newE);
    }

    let mut last_idx = 0;

    for i in 0..total_num {
        dist_start = dist_UAF_start_map[&i];
        dist_end = dist_UAF_end_map[&i];
        dist_num = dist_UAF_num_map[&i];

        let block_size = dist_end - dist_start + 1;
        last_idx = dist_start + ((dist_num as i64) * block_size) - 1;

        let mut new_gH = arrayfire::rows(gH, dist_start, last_idx);
        let mut new_gA = arrayfire::rows(gA, dist_start, last_idx);
        let mut new_gB = arrayfire::rows(gB, dist_start, last_idx);
        let mut new_gC = arrayfire::rows(gC, dist_start, last_idx);
        let mut new_gD = arrayfire::rows(gD, dist_start, last_idx);
        let mut new_gE = arrayfire::rows(gE, dist_start, last_idx);

        let new_dims = arrayfire::Dim4::new(&[block_size as u64, dist_num, 1, 1]);
        new_gH = arrayfire::moddims(&new_gH, new_dims);
        new_gA = arrayfire::moddims(&new_gA, new_dims);
        new_gB = arrayfire::moddims(&new_gB, new_dims);
        new_gC = arrayfire::moddims(&new_gC, new_dims);
        new_gD = arrayfire::moddims(&new_gD, new_dims);
        new_gE = arrayfire::moddims(&new_gE, new_dims);

        new_gH = arrayfire::mean(&new_gH, 1);
        new_gA = arrayfire::mean(&new_gA, 1);
        new_gB = arrayfire::mean(&new_gB, 1);
        new_gC = arrayfire::mean(&new_gC, 1);
        new_gD = arrayfire::mean(&new_gD, 1);
        new_gE = arrayfire::mean(&new_gE, 1);

        newH = arrayfire::rows(H, dist_start, dist_end) + (alpha0 * -new_gH);
        newA = arrayfire::rows(A, dist_start, dist_end) + (alpha1 * -new_gA);
        newB = arrayfire::rows(B, dist_start, dist_end) + (alpha1 * -new_gB);
        newC = arrayfire::rows(C, dist_start, dist_end) + (alpha1 * -new_gC);
        newD = arrayfire::rows(D, dist_start, dist_end) + (alpha1 * -new_gD);
        newE = arrayfire::rows(E, dist_start, dist_end) + (alpha1 * -new_gE);

        let tile_dims = arrayfire::Dim4::new(&[dist_num, 1, 1, 1]);
        newH = arrayfire::tile(&newH, tile_dims);
        newA = arrayfire::tile(&newA, tile_dims);
        newB = arrayfire::tile(&newB, tile_dims);
        newC = arrayfire::tile(&newC, tile_dims);
        newD = arrayfire::tile(&newD, tile_dims);
        newE = arrayfire::tile(&newE, tile_dims);

        let seqs = &[arrayfire::Seq::new(
            dist_start as f64,
            last_idx as f64,
            1.0 as f64,
        )];
        arrayfire::assign_seq(H, seqs, &newH);
        arrayfire::assign_seq(A, seqs, &newA);
        arrayfire::assign_seq(B, seqs, &newB);
        arrayfire::assign_seq(C, seqs, &newC);
        arrayfire::assign_seq(D, seqs, &newD);
        arrayfire::assign_seq(E, seqs, &newE);
    }

    let H_len = H.dims()[0] as i64;

    newH = arrayfire::rows(H, last_idx + 1, H_len - 1)
        + (alpha0 * -arrayfire::rows(gH, last_idx + 1, H_len - 1));
    newA = arrayfire::rows(A, last_idx + 1, H_len - 1)
        + (alpha1 * -arrayfire::rows(gA, last_idx + 1, H_len - 1));
    newB = arrayfire::rows(B, last_idx + 1, H_len - 1)
        + (alpha1 * -arrayfire::rows(gB, last_idx + 1, H_len - 1));
    newC = arrayfire::rows(C, last_idx + 1, H_len - 1)
        + (alpha1 * -arrayfire::rows(gC, last_idx + 1, H_len - 1));
    newD = arrayfire::rows(D, last_idx + 1, H_len - 1)
        + (alpha1 * -arrayfire::rows(gD, last_idx + 1, H_len - 1));
    newE = arrayfire::rows(E, last_idx + 1, H_len - 1)
        + (alpha1 * -arrayfire::rows(gE, last_idx + 1, H_len - 1));

    let seqs = &[arrayfire::Seq::new(
        (last_idx + 1) as f64,
        (H_len - 1) as f64,
        1.0 as f64,
    )];
    arrayfire::assign_seq(H, seqs, &newH);
    arrayfire::assign_seq(A, seqs, &newA);
    arrayfire::assign_seq(B, seqs, &newB);
    arrayfire::assign_seq(C, seqs, &newC);
    arrayfire::assign_seq(D, seqs, &newD);
    arrayfire::assign_seq(E, seqs, &newE);
}
