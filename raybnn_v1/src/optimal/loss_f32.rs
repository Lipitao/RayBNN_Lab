extern crate arrayfire;

const two: f32 = 2.0;
const one: f32 = 1.0;
const epsilon: f32 = 1.0e-20;
const epsilon2: f32 = 2.0e-20;

const zero: f32 = 0.0;
const high: f32 = 1000000.0;

pub fn softmax_cross_entropy(yhat: &arrayfire::Array<f32>, y: &arrayfire::Array<f32>) -> f32 {
    let output_size = y.dims()[0];
    let batch_size = y.dims()[1];
    let batch_size_f32 = batch_size as f32;

    let mut yhatmax = arrayfire::max(yhat, 0);
    yhatmax = arrayfire::transpose(&yhatmax, false);

    let (_, mut yidx) = arrayfire::imax(y, 0);
    yidx = arrayfire::transpose(&yidx, false);

    let mut actmax = arrayfire::flat(yhat);

    let N_dims = arrayfire::Dim4::new(&[batch_size, 1, 1, 1]);
    let repeat_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let offset = (output_size as u32) * arrayfire::iota::<u32>(N_dims, repeat_dims);

    yidx = yidx + offset;
    actmax = arrayfire::lookup(&actmax, &yidx, 0);

    let diff = yhatmax - actmax;

    let (r0, _) = arrayfire::sum_all::<f32>(&diff);

    (one / batch_size_f32) * (r0)
}

pub fn softmax_cross_entropy_grad(
    yhat: &arrayfire::Array<f32>,
    y: &arrayfire::Array<f32>,
) -> arrayfire::Array<f32> {
    let output_size = y.dims()[0];
    let batch_size = y.dims()[1];
    let batch_size_f32 = batch_size as f32;

    let mut expyhat = arrayfire::exp(yhat);
    expyhat = arrayfire::clamp(&expyhat, &zero, &high, false);

    let mut sumyhat = arrayfire::sum(&expyhat, 0);
    sumyhat = arrayfire::clamp(&sumyhat, &zero, &high, false);

    expyhat = arrayfire::div(&expyhat, &sumyhat, true);

    (one / batch_size_f32) * (expyhat - y)
}

pub fn sigmoid_cross_entropy(yhat: &arrayfire::Array<f32>, y: &arrayfire::Array<f32>) -> f32 {
    let minus = one - y.clone();
    let sigmoid = arrayfire::sigmoid(yhat) + epsilon;
    let logsigmoid = arrayfire::log(&sigmoid);
    let minussigmoid = one - sigmoid + epsilon2;
    let logminus = arrayfire::log(&minussigmoid);

    let total = -(arrayfire::mul(y, &logsigmoid, false) + arrayfire::mul(&minus, &logminus, false));
    let size: f32 = yhat.elements() as f32;
    let (r0, _) = arrayfire::sum_all::<f32>(&total);
    (one / size) * (r0 as f32)
}

pub fn weighted_sigmoid_cross_entropy(
    yhat: &arrayfire::Array<f32>,
    y: &arrayfire::Array<f32>,
    weight: f32,
) -> f32 {
    let minus = one - y.clone();
    let sigmoid = arrayfire::sigmoid(yhat) + epsilon;
    let logsigmoid = arrayfire::log(&sigmoid);
    let minussigmoid = one - sigmoid + epsilon2;
    let logminus = arrayfire::log(&minussigmoid);

    let total = -((weight * arrayfire::mul(y, &logsigmoid, false))
        + arrayfire::mul(&minus, &logminus, false));
    let size: f32 = yhat.elements() as f32;
    let (r0, _) = arrayfire::sum_all::<f32>(&total);
    (one / size) * (r0 as f32)
}

pub fn sigmoid_cross_entropy_grad(
    yhat: &arrayfire::Array<f32>,
    y: &arrayfire::Array<f32>,
) -> arrayfire::Array<f32> {
    let minus = one - y.clone();

    let yhatneg = -yhat.clone();
    let p0 = arrayfire::sigmoid(&yhatneg);
    let p1 = -arrayfire::sigmoid(&yhat);

    let size: f32 = yhat.elements() as f32;
    (-one / size) * (arrayfire::mul(y, &p0, false) + arrayfire::mul(&minus, &p1, false))
}

pub fn weighted_sigmoid_cross_entropy_grad(
    yhat: &arrayfire::Array<f32>,
    y: &arrayfire::Array<f32>,
    weight: f32,
) -> arrayfire::Array<f32> {
    let minus = one - y.clone();

    let yhatneg = -yhat.clone();
    let p0 = arrayfire::sigmoid(&yhatneg);
    let p1 = -arrayfire::sigmoid(&yhat);

    let size: f32 = yhat.elements() as f32;
    (-one / size) * ((weight * arrayfire::mul(y, &p0, false)) + arrayfire::mul(&minus, &p1, false))
}

pub fn MAE(yhat: &arrayfire::Array<f32>, y: &arrayfire::Array<f32>) -> f32 {
    let mut diff = yhat.clone() - y.clone();

    diff = arrayfire::abs(&diff);

    let (r0, _) = arrayfire::mean_all(&diff);

    r0 as f32
}

pub fn MSE(yhat: &arrayfire::Array<f32>, y: &arrayfire::Array<f32>) -> f32 {
    let diff = yhat.clone() - y.clone();
    let size: f32 = yhat.elements() as f32;

    let diff = arrayfire::pow(&diff, &two, false);
    let (r0, _) = arrayfire::sum_all::<f32>(&diff);
    (one / size) * (r0 as f32)
}

pub fn MSE_grad(yhat: &arrayfire::Array<f32>, y: &arrayfire::Array<f32>) -> arrayfire::Array<f32> {
    let size: f32 = yhat.elements() as f32;
    (two / size) * (yhat.clone() - y.clone())
}

pub fn RMSE(yhat: &arrayfire::Array<f32>, y: &arrayfire::Array<f32>) -> f32 {
    MSE(yhat, y).sqrt()
}
