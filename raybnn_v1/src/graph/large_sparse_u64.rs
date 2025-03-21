extern crate arrayfire;

pub fn COO_find(
    WRowIdxCOO: &arrayfire::Array<u64>,
    target_rows: &arrayfire::Array<u64>,
) -> arrayfire::Array<u64> {
    let target_row_num = target_rows.dims()[0];
    let WRowIdxCOO_num = WRowIdxCOO.dims()[0];
    let COO_dims = arrayfire::Dim4::new(&[1, target_row_num, 1, 1]);
    let WRowIdxCOO_tile = arrayfire::tile(WRowIdxCOO, COO_dims);
    let mut trans_rows = arrayfire::transpose(target_rows, false);
    let trans_rows_dims = arrayfire::Dim4::new(&[WRowIdxCOO_num, 1, 1, 1]);
    trans_rows = arrayfire::tile(&trans_rows, trans_rows_dims);

    let mut bool_result = arrayfire::eq(&WRowIdxCOO_tile, &trans_rows, false);

    bool_result = arrayfire::any_true(&bool_result, 1);

    arrayfire::locate(&bool_result).cast::<u64>()
}

pub fn COO_batch_find(
    WRowIdxCOO: &arrayfire::Array<u64>,
    target_rows: &arrayfire::Array<u64>,
    batch_size: u64,
) -> arrayfire::Array<u64> {
    let mut i: u64 = 0;
    let mut startseq: u64 = 0;
    let mut endseq: u64 = 0;
    let total_size = target_rows.dims()[0];

    startseq = i;
    endseq = i + batch_size - 1;
    if (endseq >= (total_size - 1)) {
        endseq = total_size - 1;
    }

    let seqs = &[arrayfire::Seq::new(startseq as u32, endseq as u32, 1)];
    let inputarr = arrayfire::index(target_rows, seqs);

    let mut total_idx = COO_find(WRowIdxCOO, &inputarr);
    i = i + batch_size;

    while i < total_size {
        startseq = i;
        endseq = i + batch_size - 1;
        if (endseq >= (total_size - 1)) {
            endseq = total_size - 1;
        }

        let seqs = &[arrayfire::Seq::new(startseq as u32, endseq as u32, 1)];
        let inputarr = arrayfire::index(target_rows, seqs);

        let idx = COO_find(WRowIdxCOO, &inputarr);

        if (idx.dims()[0] > 0) {
            total_idx = arrayfire::join(0, &total_idx, &idx);
        }

        i = i + batch_size;
    }

    arrayfire::sort(&total_idx, 0, true)
}
