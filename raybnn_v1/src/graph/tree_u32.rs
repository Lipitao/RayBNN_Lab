extern crate arrayfire;

pub fn find_unique(arr: &arrayfire::Array<u32>, neuron_size: u64) -> arrayfire::Array<u32> {
    let table_dims = arrayfire::Dim4::new(&[neuron_size, 1, 1, 1]);
    let mut table = arrayfire::constant::<bool>(false, table_dims);

    let inarr = arrayfire::constant::<bool>(true, arr.dims());
    //let idxarr = arr.cast::<u32>();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(arr, 0, None);
    arrayfire::assign_gen(&mut table, &idxrs, &inarr);

    arrayfire::locate(&table).cast::<u32>()
}

pub fn parallel_lookup(
    batch_dim: u64,
    lookup_dim: u64,

    idx: &arrayfire::Array<u32>,
    target: &arrayfire::Array<u32>,
) -> arrayfire::Array<u32> {
    let target_dims = target.dims();

    let batch_num = target_dims[batch_dim as usize];
    let lookup_size = target_dims[lookup_dim as usize];

    let repeat_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let mut tile_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    tile_dims[batch_dim as usize] = batch_num;

    let count = arrayfire::iota::<u32>(tile_dims, repeat_dims);

    let mut idx2 = batch_num * idx.clone();

    idx2 = arrayfire::add(&idx2, &count, true);

    drop(count);

    idx2 = arrayfire::flat(&idx2);

    let mut ouput_arr = arrayfire::flat(target);

    ouput_arr = arrayfire::lookup(&ouput_arr, &idx2, 0);

    drop(idx2);

    arrayfire::moddims(&ouput_arr, idx.dims())
}
