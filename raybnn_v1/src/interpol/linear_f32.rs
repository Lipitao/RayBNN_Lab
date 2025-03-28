extern crate arrayfire;

//t: times values of function f
//f: values of function f at time t
//dfdt:  derivatives of function f at time t
//s: time vector to sample
//Output: values of function f at time s
pub fn find(
    t: &arrayfire::Array<f32>,
    f: &arrayfire::Array<f32>,
    dfdt: &arrayfire::Array<f32>,
    s: &arrayfire::Array<f32>,
) -> arrayfire::Array<f32> {
    let s_dims = s.dims();
    let s_num = s_dims[0];

    let t_dims = t.dims();
    let t_num = t_dims[0];

    let f_dims = f.dims();
    let f_num = f_dims[1];

    let t_dims = arrayfire::Dim4::new(&[1, s_num, 1, 1]);
    let t_arr = arrayfire::tile(&t, t_dims);

    let s_arr = arrayfire::transpose(&s, false);
    let s_dims = arrayfire::Dim4::new(&[t_num, 1, 1, 1]);
    let s_arr = arrayfire::tile(&s_arr, s_dims);

    let dist = s_arr - t_arr;

    let dist = arrayfire::abs(&dist);
    let (_, idx) = arrayfire::imin(&dist, 0);
    drop(dist);

    let seq4gen = arrayfire::Seq::new(0.0, (f_num - 1) as f32, 1.0);
    let mut t_idxer = arrayfire::Indexer::default();
    t_idxer.set_index(&idx, 0, None);

    let mut f_idxer = arrayfire::Indexer::default();
    f_idxer.set_index(&idx, 0, None);
    f_idxer.set_index(&seq4gen, 1, Some(false));

    let mut dfdt_idxer = arrayfire::Indexer::default();
    dfdt_idxer.set_index(&idx, 0, None);
    dfdt_idxer.set_index(&seq4gen, 1, Some(false));

    let t_init = arrayfire::index_gen(&t, t_idxer);
    let f_init = arrayfire::index_gen(&f, f_idxer);
    let dfdt_init = arrayfire::index_gen(&dfdt, dfdt_idxer);

    let step = s - t_init;

    let step_dims = arrayfire::Dim4::new(&[1, f_num, 1, 1]);
    let step = arrayfire::tile(&step, step_dims);

    let result = f_init + arrayfire::mul(&dfdt_init, &step, false);

    result
}
