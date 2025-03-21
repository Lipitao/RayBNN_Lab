extern crate arrayfire;

pub fn find_unique(arr: &arrayfire::Array<u64>, neuron_size: u64) -> arrayfire::Array<u64> {
    let table_dims = arrayfire::Dim4::new(&[neuron_size, 1, 1, 1]);
    let mut table = arrayfire::constant::<bool>(false, table_dims);

    let inarr = arrayfire::constant::<bool>(true, arr.dims());
    //let idxarr = arr.cast::<u64>();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(arr, 0, None);
    arrayfire::assign_gen(&mut table, &idxrs, &inarr);

    arrayfire::locate(&table).cast::<u64>()
}
