extern crate arrayfire;

const two: f64 = 2.0;
const one: f64 = 1.0;
const epsilon: f64 = 1.0e-8;

pub fn adam(
    beta0: f64,
    beta1: f64,
    direction: &mut arrayfire::Array<f64>,
    mt: &mut arrayfire::Array<f64>,
    vt: &mut arrayfire::Array<f64>,
) {
    *mt = (mt.clone()) * beta0 + (one - beta0) * (direction.clone());
    *vt = (vt.clone()) * beta1 + (one - beta1) * arrayfire::pow(direction, &two, false);

    let nmt = mt.clone() / (one - beta0);
    let mut nvt = vt.clone() / (one - beta1);
    nvt = arrayfire::sqrt(&nvt) + epsilon;

    *direction = (nmt / nvt);
}

pub fn momentum(beta: f64, grad: &arrayfire::Array<f64>, dir: &mut arrayfire::Array<f64>) {
    *dir = (dir.clone() * beta) + (one - beta) * (grad.clone());
}
