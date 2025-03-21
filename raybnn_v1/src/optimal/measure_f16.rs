use half;
use num::traits::real::Real;
extern crate arrayfire;

use crate::optimal::measure_u32::confusion_matrix;

const two: half::f16 = half::f16::from_f32_const(2.0f32);

pub fn precision_recall_f1_MCC_binary(
    yhat: &arrayfire::Array<u32>,
    y: &arrayfire::Array<u32>,
) -> arrayfire::Array<half::f16> {
    let con_matrix = confusion_matrix(yhat, y, 2).cast::<half::f16>();

    let mut con_matrix_cpu = vec![half::f16::default(); con_matrix.elements()];

    con_matrix.host(&mut con_matrix_cpu);

    let TP = con_matrix_cpu[3];
    let FP = con_matrix_cpu[2];
    let FN = con_matrix_cpu[1];
    let TN = con_matrix_cpu[0];

    let P = TP / (TP + FP);
    let R = TP / (TP + FN);

    let result: Vec<half::f16> = vec![
        P,
        R,
        two * ((P * R) / (P + R)),
        (TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)).sqrt()),
    ];

    arrayfire::Array::new(
        &result,
        arrayfire::Dim4::new(&[result.len() as u64, 1, 1, 1]),
    )
}

pub fn precision_recall_f1_MCC_multi(
    yhat: &arrayfire::Array<u32>,
    y: &arrayfire::Array<u32>,
    weights: &arrayfire::Array<half::f16>,
    label_num: u64,
) -> arrayfire::Array<half::f16> {
    let mut count: u32 = 0;

    let mut temp_yhat = arrayfire::eq(yhat, &count, false).cast::<u32>();

    let mut temp_y = arrayfire::eq(y, &count, false).cast::<u32>();

    let mut result = precision_recall_f1_MCC_binary(&temp_yhat, &temp_y);

    count = count + 1;

    while count < (label_num as u32) {
        temp_yhat = arrayfire::eq(yhat, &count, false).cast::<u32>();

        temp_y = arrayfire::eq(y, &count, false).cast::<u32>();

        let temp_result = precision_recall_f1_MCC_binary(&temp_yhat, &temp_y);

        result = arrayfire::join(1, &result, &temp_result);

        count = count + 1;
    }

    arrayfire::mean_weighted(&result, weights, 1)
}
