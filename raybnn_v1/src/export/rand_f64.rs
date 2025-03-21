extern crate arrayfire;

pub fn single_random_uniform() -> f64 {
    let single_rand_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let singlerand = arrayfire::randu::<f64>(single_rand_dims);

    let mut singlerand_cpu: [f64; 1] = [0.0];
    singlerand.host(&mut singlerand_cpu);

    return singlerand_cpu[0];
}
