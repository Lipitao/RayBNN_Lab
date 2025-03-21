extern crate arrayfire;

use rayon::prelude::*;

fn gen_const(pair: (usize, i32)) -> Vec<i32> {
    let (i, e) = pair;
    let a: Vec<i32> = vec![i as i32; e as usize];
    a
}

pub fn remap_rows(
    rowvec: &arrayfire::Array<i32>,
    idxsel: &arrayfire::Array<i32>,
    row_num: u64,
) -> arrayfire::Array<i32> {
    let table_dims = arrayfire::Dim4::new(&[row_num, 1, 1, 1]);
    let mut table = arrayfire::constant::<i32>(0, table_dims);

    let single = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut indexarr = arrayfire::iota::<i32>(idxsel.dims(), single);

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(idxsel, 0, None);
    arrayfire::assign_gen(&mut table, &idxrs, &indexarr);

    arrayfire::lookup(&table, rowvec, 0)
}

pub fn COO_find(
    WRowIdxCOO: &arrayfire::Array<i32>,
    target_rows: &arrayfire::Array<i32>,
) -> arrayfire::Array<i32> {
    let target_row_num = target_rows.dims()[0];
    let WRowIdxCOO_num = WRowIdxCOO.dims()[0];
    let COO_dims = arrayfire::Dim4::new(&[1, target_row_num, 1, 1]);
    let WRowIdxCOO_tile = arrayfire::tile(WRowIdxCOO, COO_dims);
    let mut trans_rows = arrayfire::transpose(target_rows, false);
    let trans_rows_dims = arrayfire::Dim4::new(&[WRowIdxCOO_num, 1, 1, 1]);
    trans_rows = arrayfire::tile(&trans_rows, trans_rows_dims);

    let mut bool_result = arrayfire::eq(&WRowIdxCOO_tile, &trans_rows, false);

    bool_result = arrayfire::any_true(&bool_result, 1);

    arrayfire::locate(&bool_result).cast::<i32>()
}

pub fn COO_batch_find(
    WRowIdxCOO: &arrayfire::Array<i32>,
    target_rows: &arrayfire::Array<i32>,
    batch_size: u64,
) -> arrayfire::Array<i32> {
    let mut i: u64 = 0;
    let mut startseq: u64 = 0;
    let mut endseq: u64 = 0;
    let total_size = target_rows.dims()[0];

    startseq = i;
    endseq = i + batch_size - 1;
    if (endseq >= (total_size - 1)) {
        endseq = total_size - 1;
    }

    let seqs = &[arrayfire::Seq::new(startseq as i32, endseq as i32, 1)];
    let inputarr = arrayfire::index(target_rows, seqs);

    let mut total_idx = COO_find(WRowIdxCOO, &inputarr);
    i = i + batch_size;

    while i < total_size {
        startseq = i;
        endseq = i + batch_size - 1;
        if (endseq >= (total_size - 1)) {
            endseq = total_size - 1;
        }

        let seqs = &[arrayfire::Seq::new(startseq as i32, endseq as i32, 1)];
        let inputarr = arrayfire::index(target_rows, seqs);

        let idx = COO_find(WRowIdxCOO, &inputarr);

        if (idx.dims()[0] > 0) {
            total_idx = arrayfire::join(0, &total_idx, &idx);
        }

        i = i + batch_size;
    }

    arrayfire::sort(&total_idx, 0, true)
}

pub fn COO_to_CSR(WRowIdxCOO: &arrayfire::Array<i32>, row_num: u64) -> arrayfire::Array<i32> {
    let WRowIdxCOO_num = WRowIdxCOO.dims()[0];

    let ones = arrayfire::constant::<i32>(1, arrayfire::Dim4::new(&[WRowIdxCOO_num, 1, 1, 1]));
    let mut temparr = arrayfire::constant::<i32>(0, arrayfire::Dim4::new(&[row_num, 1, 1, 1]));

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(WRowIdxCOO, 0, None);
    arrayfire::assign_gen(&mut temparr, &idxrs, &ones);

    let sel = arrayfire::locate(&temparr);

    //let  (_,mut sumarr) = arrayfire::count_by_key(WRowIdxCOO, &ones, 0);
    let (_, mut sumarr) = arrayfire::sum_by_key(WRowIdxCOO, &ones, 0);

    sumarr = sumarr.cast::<i32>();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&sel, 0, None);
    arrayfire::assign_gen(&mut temparr, &idxrs, &sumarr);

    temparr = arrayfire::accum(&temparr, 0);

    let constarr = arrayfire::constant::<i32>(0, arrayfire::Dim4::new(&[1, 1, 1, 1]));
    temparr = arrayfire::join(0, &constarr, &temparr);

    temparr
}

pub fn CSR_to_COO(WRowIdxCSR: &arrayfire::Array<i32>) -> arrayfire::Array<i32> {
    let rsize: u64 = WRowIdxCSR.dims()[0];
    let r0 = arrayfire::rows(WRowIdxCSR, 0, (rsize - 2) as i64);
    let r1 = arrayfire::rows(WRowIdxCSR, 1, (rsize - 1) as i64);

    let rowdiff = r1.clone() - r0.clone();

    let mut rowdiff_cpu = vec![i32::default(); rowdiff.elements()];
    rowdiff.host(&mut rowdiff_cpu);

    let (count, _) = arrayfire::sum_all::<i32>(&rowdiff);

    let WRowIdxCOO_dims = arrayfire::Dim4::new(&[count as u64, 1, 1, 1]);

    let mut WRowIdxCOO_cpu: Vec<i32> = rowdiff_cpu
        .into_par_iter()
        .enumerate()
        .map(gen_const)
        .flatten_iter()
        .collect();

    arrayfire::Array::new(&WRowIdxCOO_cpu, WRowIdxCOO_dims)
}

pub fn integer_histogram(
    input: &arrayfire::Array<i32>,

    bins: &mut arrayfire::Array<i32>,
    counts: &mut arrayfire::Array<u32>,
) {
    let sorted = arrayfire::sort(&input, 0, true);

    let ones = arrayfire::constant::<i32>(1, sorted.dims());
    let (keys, values) = arrayfire::sum_by_key(&sorted, &ones, 0);

    *bins = keys;
    *counts = values.cast::<u32>();
}
