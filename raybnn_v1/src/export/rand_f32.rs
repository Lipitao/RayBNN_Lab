extern crate arrayfire;

pub fn single_random_uniform() -> f32 {
    let single_rand_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let singlerand = arrayfire::randu::<f32>(single_rand_dims);

    let mut singlerand_cpu: [f32; 1] = [0.0];
    singlerand.host(&mut singlerand_cpu);

    return singlerand_cpu[0];
}
