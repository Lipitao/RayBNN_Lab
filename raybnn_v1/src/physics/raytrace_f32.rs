extern crate arrayfire;
use num::integer::Roots;
use rayon::prelude::*;

use rand::prelude::*;

use crate::graph::adjacency_f32::get_global_weight_idx;

use crate::neural::network_f32::network_metadata_type;

use crate::graph::tree_i32::find_unique;

use serde::{Deserialize, Serialize};

use nohash_hasher;
use rand::distributions::{Distribution, Uniform};
use std::collections::HashMap;

const two: f32 = 2.0;
const one: f32 = 1.0;
const zero: f32 = 0.0;

const epsilon: f32 = 1.0e-3;

const oneminuseps: f32 = one - epsilon;

const AVG_RAYTRACE_NUM: u64 = 500;
const RAYTRACE_LIMIT: u64 = 100000000;

const COUNT_LIMIT: u64 = 10000000000;

const epsilon2: f32 = 1.0e-8;

#[derive(Serialize, Deserialize)]
pub struct raytrace_option_type {
    pub max_rounds: u64,
    pub input_connection_num: u64,
    pub ray_neuron_intersect: bool,
    pub ray_glia_intersect: bool,
}

pub fn generate_random_rays_to_center(
    neuron_pos: &arrayfire::Array<f32>,
    ray_num: u64,
    con_rad: f32,

    start_line: &mut arrayfire::Array<f32>,
    dir_line: &mut arrayfire::Array<f32>,
) {
    let space_dims: u64 = neuron_pos.dims()[1];

    let tile_dims = arrayfire::Dim4::new(&[ray_num, 1, 1, 1]);

    *start_line = arrayfire::tile(neuron_pos, tile_dims);

    *dir_line = start_line.clone() * -1.0f32;

    //Mag of dir_line
    let mut mag2 = arrayfire::pow(dir_line, &two, false);
    mag2 = arrayfire::sum(&mag2, 1);

    //Generate random vectors
    let start_line_num = start_line.dims()[0];
    let rand_dims = arrayfire::Dim4::new(&[start_line_num, space_dims, 1, 1]);
    let mut rand_vec = (arrayfire::randu::<f32>(rand_dims) - 0.5f32);

    //Normalize random Vector
    let mut mag = arrayfire::pow(&rand_vec, &two, false);
    mag = arrayfire::sum(&mag, 1);
    mag = arrayfire::sqrt(&mag) + epsilon2;

    //Scale random vector to connection radius
    rand_vec = arrayfire::div(&rand_vec, &mag, true);
    mag = arrayfire::sqrt(&mag2);
    rand_vec = arrayfire::mul(&rand_vec, &mag, true);
    drop(mag);

    //Vector Projection
    let mut projvec = arrayfire::mul(&rand_vec, dir_line, false);
    projvec = arrayfire::sum(&projvec, 1);

    mag2 = mag2 + epsilon2;
    projvec = arrayfire::div(&projvec, &mag2, false);
    drop(mag2);

    //Vector rejection
    projvec = rand_vec.clone() - arrayfire::mul(&projvec, dir_line, true);
    drop(rand_vec);

    //Random scale
    let rand2_dims = arrayfire::Dim4::new(&[start_line_num, 1, 1, 1]);
    let mut rand2_vec = 2.0f32 * arrayfire::randu::<f32>(rand2_dims);
    projvec = arrayfire::mul(&projvec, &rand2_vec, true);

    *dir_line = dir_line.clone() + projvec;

    //Scale dir line
    let mut mag3 = arrayfire::pow(dir_line, &two, false);
    mag3 = arrayfire::sum(&mag3, 1);
    mag3 = arrayfire::sqrt(&mag3) + epsilon2;

    *dir_line = con_rad * arrayfire::div(dir_line, &mag3, true);
}

pub fn generate_random_uniform_rays(
    neuron_pos: &arrayfire::Array<f32>,
    ray_num: u64,
    con_rad: f32,

    start_line: &mut arrayfire::Array<f32>,
    dir_line: &mut arrayfire::Array<f32>,
) {
    let space_dims: u64 = neuron_pos.dims()[1];

    let tile_dims = arrayfire::Dim4::new(&[ray_num, 1, 1, 1]);

    *start_line = arrayfire::tile(neuron_pos, tile_dims);

    if space_dims == 2 {
        let start_line_num = start_line.dims()[0];
        let t_dims = arrayfire::Dim4::new(&[start_line_num, 1, 1, 1]);
        let t = two * std::f32::consts::PI * arrayfire::randu::<f32>(t_dims);

        let x = con_rad * arrayfire::cos(&t);
        let y = con_rad * arrayfire::sin(&t);

        *dir_line = arrayfire::join(1, &x, &y);
    } else {
        let start_line_num = start_line.dims()[0];
        let t_dims = arrayfire::Dim4::new(&[start_line_num, 1, 1, 1]);
        let mut t = two * std::f32::consts::PI * arrayfire::randu::<f32>(t_dims);

        *dir_line = con_rad * arrayfire::cos(&t);
        for i in 1..(space_dims - 1) {
            let mut newd = arrayfire::sin(&t);

            let newt = two * std::f32::consts::PI * arrayfire::randu::<f32>(t_dims);
            let lastd = arrayfire::cos(&newt);
            newd = arrayfire::join(1, &newd, &lastd);
            newd = con_rad * arrayfire::product(&newd, 1);

            *dir_line = arrayfire::join(1, dir_line, &newd);
            t = arrayfire::join(1, &t, &newt);
        }

        //let newt = two*std::f32::consts::PI*arrayfire::randu::<f32>(t_dims);
        //t = arrayfire::join(1, &t, &newt);
        let mut newd = arrayfire::sin(&t);
        newd = con_rad * arrayfire::product(&newd, 1);
        *dir_line = arrayfire::join(1, dir_line, &newd);
    }
}

pub fn tileDown(repeat_num: u64, input_arr: &mut arrayfire::Array<f32>) {
    let space_dims: u64 = input_arr.dims()[1];

    let input_arr_num: u64 = input_arr.dims()[0];

    let tile_dims = arrayfire::Dim4::new(&[1, repeat_num, 1, 1]);

    *input_arr = arrayfire::tile(input_arr, tile_dims);

    *input_arr = arrayfire::transpose(input_arr, false);

    let dims = arrayfire::Dim4::new(&[space_dims, repeat_num * input_arr_num, 1, 1]);
    *input_arr = arrayfire::moddims(input_arr, dims);

    *input_arr = arrayfire::transpose(input_arr, false);
}

pub fn tileDown_i32(repeat_num: u64, input_arr: &mut arrayfire::Array<i32>) {
    let space_dims: u64 = input_arr.dims()[1];

    let input_arr_num: u64 = input_arr.dims()[0];

    let tile_dims = arrayfire::Dim4::new(&[1, repeat_num, 1, 1]);

    *input_arr = arrayfire::tile(input_arr, tile_dims);

    *input_arr = arrayfire::transpose(input_arr, false);

    let dims = arrayfire::Dim4::new(&[space_dims, repeat_num * input_arr_num, 1, 1]);
    *input_arr = arrayfire::moddims(input_arr, dims);

    *input_arr = arrayfire::transpose(input_arr, false);
}

pub fn filter_rays(
    con_rad: f32,

    target_input_pos: &arrayfire::Array<f32>,

    input_pos: &mut arrayfire::Array<f32>,
    input_idx: &mut arrayfire::Array<i32>,
) {
    let input_diff = arrayfire::sub(target_input_pos, input_pos, true);

    let con_rad_sq = con_rad * con_rad;

    let mut mag2 = arrayfire::pow(&input_diff, &two, false);
    mag2 = arrayfire::sum(&mag2, 1);

    //  (con_rad_sq >= mag2 )
    let CMPRET = arrayfire::ge(&con_rad_sq, &mag2, false);
    drop(mag2);

    //Lookup  1 >= dir_line  >= 0
    let idx_intersect = arrayfire::locate(&CMPRET);
    drop(CMPRET);

    *input_pos = arrayfire::lookup(input_pos, &idx_intersect, 0);

    *input_idx = arrayfire::lookup(input_idx, &idx_intersect, 0);
}

pub fn rays_from_neuronsA_to_neuronsB(
    con_rad: f32,

    neuronA_pos: &arrayfire::Array<f32>,
    neuronB_pos: &arrayfire::Array<f32>,

    start_line: &mut arrayfire::Array<f32>,
    dir_line: &mut arrayfire::Array<f32>,

    input_idx: &mut arrayfire::Array<i32>,
    hidden_idx: &mut arrayfire::Array<i32>,
) {
    let space_dims: u64 = neuronA_pos.dims()[1];

    let neuronA_num: u64 = neuronA_pos.dims()[0];
    let neuronB_num: u64 = neuronB_pos.dims()[0];

    let tile_dims = arrayfire::Dim4::new(&[neuronB_num, 1, 1, 1]);

    *start_line = arrayfire::tile(neuronA_pos, tile_dims);

    *dir_line = neuronB_pos.clone();

    /*
    let tile_dims = arrayfire::Dim4::new(&[1,neuronA_num,1,1]);

    *dir_line = arrayfire::tile(neuronB_pos, tile_dims);

    *dir_line = arrayfire::transpose(dir_line, false);

    let dims = arrayfire::Dim4::new(&[space_dims, neuronA_num*neuronB_num , 1 , 1]);
    *dir_line = arrayfire::moddims(dir_line, dims);

    *dir_line = arrayfire::transpose(dir_line, false);

    */
    tileDown(neuronA_num, dir_line);

    *dir_line = dir_line.clone() - start_line.clone();

    let con_rad_sq = con_rad * con_rad;

    let mut mag2 = arrayfire::pow(dir_line, &two, false);
    mag2 = arrayfire::sum(&mag2, 1);

    //  (con_rad_sq >= mag2 )
    let CMPRET = arrayfire::ge(&con_rad_sq, &mag2, false);
    drop(mag2);

    //Lookup  1 >= dir_line  >= 0
    let idx_intersect = arrayfire::locate(&CMPRET);
    drop(CMPRET);

    *start_line = arrayfire::lookup(start_line, &idx_intersect, 0);

    *dir_line = arrayfire::lookup(dir_line, &idx_intersect, 0);

    *input_idx = arrayfire::lookup(input_idx, &idx_intersect, 0);

    *hidden_idx = arrayfire::lookup(hidden_idx, &idx_intersect, 0);
}

pub fn line_sphere_intersect(
    start_line: &arrayfire::Array<f32>,
    dir_line: &arrayfire::Array<f32>,

    circle_center: &arrayfire::Array<f32>,
    circle_radius: &arrayfire::Array<f32>,

    intersect: &mut arrayfire::Array<bool>,
) {
    let line_num: u64 = start_line.dims()[0];

    let circle_num: u64 = circle_center.dims()[0];

    let space_dims = start_line.dims()[1];

    // C^T
    let mut CENTERSUBSTART = arrayfire::reorder_v2(&circle_center, 2, 1, Some(vec![0]));

    // C - S
    CENTERSUBSTART = arrayfire::sub(&CENTERSUBSTART, start_line, true);
    //drop(circle_center_trans);

    // dot(C - S, D)
    let mut dotret = arrayfire::mul(&CENTERSUBSTART, dir_line, true);

    dotret = arrayfire::sum(&dotret, 1);

    // |D|^2
    let mut sq = arrayfire::pow(dir_line, &two, false);
    sq = arrayfire::sum(&sq, 1);

    // dot(C - S, D)  /  |D|^2
    dotret = arrayfire::div(&dotret, &sq, true);
    drop(sq);

    // Clamp(     dot(C - S, D)  /  |D|^2      )
    dotret = arrayfire::clamp(&dotret, &zero, &one, false);

    // Clamp(     dot(C - S, D)  /  |D|^2      )   D
    dotret = arrayfire::mul(&dotret, dir_line, true);

    // (C - S)   -   Clamp( dot(C - S, D)  /  |D|^2  ) D
    dotret = CENTERSUBSTART - dotret;

    // Mag( Vector Rejection )
    dotret = arrayfire::pow(&dotret, &two, false);
    dotret = arrayfire::sum(&dotret, 1);

    // R^T
    let mut tempradius = arrayfire::reorder_v2(&circle_radius, 2, 1, Some(vec![0]));

    // R^2
    tempradius = arrayfire::pow(&tempradius, &two, false);

    //  (tempradius >= tempdir )
    *intersect = arrayfire::ge(&tempradius, &dotret, true);
}

/*

pub fn line_sphere_intersect(
    start_line: &arrayfire::Array<f32>,
    dir_line: &arrayfire::Array<f32>,

    circle_center: &arrayfire::Array<f32>,
    circle_radius: &arrayfire::Array<f32>,

    intersect: &mut arrayfire::Array<bool>
    )
{
    let line_num: u64 = start_line.dims()[0];

    let circle_num: u64 = circle_center.dims()[0];

    let space_dims = start_line.dims()[1];




    // C^T
    let circle_center_trans = arrayfire::reorder_v2(&circle_center, 2, 1, Some(vec![0]));


    // C - S
    let mut CENTERSUBSTART = arrayfire::sub(&circle_center_trans,start_line,true);

    // dot(C - S, D)
    let mut dotret = arrayfire::mul(&CENTERSUBSTART,dir_line,true);

    dotret = arrayfire::sum(&dotret,1);



    // |D|^2
    let mut sq = arrayfire::pow(dir_line,&two,false);
    sq = arrayfire::sum(&sq, 1);




    // dot(C - S, D)  /  |D|^2
    dotret = arrayfire::div(&dotret,&sq,true);


    //  (dotret >= 0 )
    let mut CMPRET = arrayfire::ge(&dotret, &zero, false);




    //Lookup  1 >= dotret  >= 0
    let idx_intersect = arrayfire::locate(&CMPRET);


    *intersect =  CMPRET.clone();


    //if empty quit
    if (idx_intersect.dims()[0] == 0)
    {
        return;
    }

    //Flatten dotret
    dotret =  arrayfire::flat(&dotret);

    //Lookup  dotret
    dotret = arrayfire::lookup(&dotret, &idx_intersect, 0);


    dotret = arrayfire::clamp(&dotret, &zero, &one, false);






    let temp_idx =   arrayfire::modulo(&idx_intersect,&line_num,false);

    //Lookup
    let mut tempdir = arrayfire::lookup(dir_line, &temp_idx, 0);





    let temp_idx2 =   arrayfire::div(&idx_intersect,&line_num,false);

    CENTERSUBSTART = arrayfire::reorder_v2(&CENTERSUBSTART, 0, 2, Some(vec![1]));

    let CENTERSUBSTART_dims = arrayfire::Dim4::new(&[line_num*circle_num,space_dims,1,1]);
    CENTERSUBSTART = arrayfire::moddims(&CENTERSUBSTART,CENTERSUBSTART_dims);

    let tempcenter = arrayfire::lookup(&CENTERSUBSTART, &idx_intersect, 0);

    let mut tempradius = arrayfire::lookup(&circle_radius, &temp_idx2, 0);


    // Project C - S onto D
    tempdir = arrayfire::mul(&tempdir, &dotret, true);





    // Project  - tempcenter
    tempdir = tempdir - tempcenter;



    tempdir = arrayfire::pow(&tempdir,&two,false);
    tempdir = arrayfire::sum(&tempdir, 1);



    tempradius = arrayfire::pow(&tempradius,&two,false);


    //  (tempradius >= tempdir )
    CMPRET = arrayfire::ge(&tempradius, &tempdir, false);




    let mut newidx = arrayfire::locate(&CMPRET);
    drop(CMPRET);

    //Lookup
    newidx = arrayfire::lookup(&idx_intersect, &newidx, 0);




    let intersect_dims = arrayfire::Dim4::new(&[line_num*circle_num,1,1,1]);
    *intersect = arrayfire::constant::<bool>(false,intersect_dims);



    let inarr = arrayfire::constant::<bool>(true, newidx.dims());

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&newidx, 0, None);
    arrayfire::assign_gen(intersect, &idxrs, &inarr);



    let intersect_dims = arrayfire::Dim4::new(&[line_num,1,circle_num,1]);
    *intersect = arrayfire::moddims(intersect,intersect_dims);


}

*/

pub fn line_sphere_intersect_batch(
    batch_size: u64,
    start_line: &arrayfire::Array<f32>,
    dir_line: &arrayfire::Array<f32>,

    circle_center: &arrayfire::Array<f32>,
    circle_radius: &arrayfire::Array<f32>,

    intersect: &mut arrayfire::Array<bool>,
) {
    let mut i: u64 = 0;
    let mut startseq: u64 = 0;
    let mut endseq: u64 = 0;
    let total_size = circle_radius.dims()[0];

    startseq = i;
    endseq = i + batch_size - 1;
    if (endseq >= (total_size - 1)) {
        endseq = total_size - 1;
    }

    let seqs = &[arrayfire::Seq::new(
        startseq as f32,
        endseq as f32,
        1.0 as f32,
    )];
    let input_circle_radius = arrayfire::index(circle_radius, seqs);

    let seqs2 = &[
        arrayfire::Seq::new(startseq as f32, endseq as f32, 1.0 as f32),
        arrayfire::Seq::default(),
    ];
    let input_circle_center = arrayfire::index(circle_center, seqs2);

    line_sphere_intersect(
        start_line,
        dir_line,
        &input_circle_center,
        &input_circle_radius,
        intersect,
    );

    i = i + batch_size;

    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let mut intersect_temp = arrayfire::constant::<bool>(false, single_dims);

    while i < total_size {
        startseq = i;
        endseq = i + batch_size - 1;
        if (endseq >= (total_size - 1)) {
            endseq = total_size - 1;
        }

        let seqs = &[arrayfire::Seq::new(
            startseq as f32,
            endseq as f32,
            1.0 as f32,
        )];
        let input_circle_radius = arrayfire::index(circle_radius, seqs);

        let seqs2 = &[
            arrayfire::Seq::new(startseq as f32, endseq as f32, 1.0 as f32),
            arrayfire::Seq::default(),
        ];
        let input_circle_center = arrayfire::index(circle_center, seqs2);

        line_sphere_intersect(
            start_line,
            dir_line,
            &input_circle_center,
            &input_circle_radius,
            &mut intersect_temp,
        );

        *intersect = arrayfire::join(2, intersect, &intersect_temp);

        i = i + batch_size;
    }
}

pub fn line_sphere_intersect_batchV2(
    batch_size: u64,

    threshold: u32,

    circle_center: &arrayfire::Array<f32>,
    circle_radius: &arrayfire::Array<f32>,

    start_line: &mut arrayfire::Array<f32>,
    dir_line: &mut arrayfire::Array<f32>,

    input_idx: &mut arrayfire::Array<i32>,
    hidden_idx: &mut arrayfire::Array<i32>,
) {
    let mut i: u64 = 0;
    let mut startseq: u64 = 0;
    let mut endseq: u64 = 0;
    let total_size = circle_radius.dims()[0];

    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let mut intersect = arrayfire::constant::<bool>(false, single_dims);

    let ray_num = start_line.dims()[0];
    let counter_dims = arrayfire::Dim4::new(&[ray_num, 1, 1, 1]);
    let mut counter = arrayfire::constant::<u32>(0, counter_dims);

    startseq = i;
    endseq = i + batch_size - 1;
    if (endseq >= (total_size - 1)) {
        endseq = total_size - 1;
    }

    let seqs = &[arrayfire::Seq::new(
        startseq as f32,
        endseq as f32,
        1.0 as f32,
    )];
    let input_circle_radius = arrayfire::index(circle_radius, seqs);

    let seqs2 = &[
        arrayfire::Seq::new(startseq as f32, endseq as f32, 1.0 as f32),
        arrayfire::Seq::default(),
    ];
    let input_circle_center = arrayfire::index(circle_center, seqs2);

    line_sphere_intersect(
        start_line,
        dir_line,
        &input_circle_center,
        &input_circle_radius,
        &mut intersect,
    );

    let mut counter_temp = intersect.cast::<u8>();
    //intersect = arrayfire::constant::<bool>(false,single_dims);
    counter = counter + arrayfire::sum(&counter_temp, 2);

    /*
    //  (threshold >= counter )
    let mut CMPRET = arrayfire::ge(&threshold, &counter, false);
    //Lookup  1 >= dir_line  >= 0
    let mut idx_intersect = arrayfire::locate(&CMPRET);

    counter = arrayfire::lookup(&counter, &idx_intersect, 0);
    *start_line = arrayfire::lookup(start_line, &idx_intersect, 0);
    *dir_line = arrayfire::lookup(dir_line, &idx_intersect, 0);
    *input_idx = arrayfire::lookup(input_idx, &idx_intersect, 0);
    *hidden_idx = arrayfire::lookup(hidden_idx, &idx_intersect, 0);
    */

    i = i + batch_size;

    let mut period = 1 + (COUNT_LIMIT / (intersect.elements() as u64));
    let mut clean_counter = 1;
    let mut CMPRET = arrayfire::constant::<bool>(false, single_dims);
    let mut idx_intersect = arrayfire::constant::<u32>(0, single_dims);

    while i < total_size {
        startseq = i;
        endseq = i + batch_size - 1;
        if (endseq >= (total_size - 1)) {
            endseq = total_size - 1;
        }

        let seqs = &[arrayfire::Seq::new(
            startseq as f32,
            endseq as f32,
            1.0 as f32,
        )];
        let input_circle_radius = arrayfire::index(circle_radius, seqs);

        let seqs2 = &[
            arrayfire::Seq::new(startseq as f32, endseq as f32, 1.0 as f32),
            arrayfire::Seq::default(),
        ];
        let input_circle_center = arrayfire::index(circle_center, seqs2);

        line_sphere_intersect(
            start_line,
            dir_line,
            &input_circle_center,
            &input_circle_radius,
            &mut intersect,
        );

        counter_temp = intersect.cast::<u8>();
        //intersect = arrayfire::constant::<bool>(false,single_dims);
        counter = counter + arrayfire::sum(&counter_temp, 2);

        clean_counter = clean_counter + 1;
        if clean_counter >= period {
            //  (threshold >= counter )
            CMPRET = arrayfire::ge(&threshold, &counter, false);
            //Lookup  1 >= dir_line  >= 0
            idx_intersect = arrayfire::locate(&CMPRET);

            counter = arrayfire::lookup(&counter, &idx_intersect, 0);
            *start_line = arrayfire::lookup(start_line, &idx_intersect, 0);
            *dir_line = arrayfire::lookup(dir_line, &idx_intersect, 0);
            *input_idx = arrayfire::lookup(input_idx, &idx_intersect, 0);
            *hidden_idx = arrayfire::lookup(hidden_idx, &idx_intersect, 0);

            period = 1 + (COUNT_LIMIT / (intersect.elements() as u64));
            clean_counter = 1;
            CMPRET = arrayfire::constant::<bool>(false, single_dims);
            idx_intersect = arrayfire::constant::<u32>(0, single_dims);
        }

        // *intersect = arrayfire::join(2, intersect, &intersect_temp);

        i = i + batch_size;
    }

    //  (threshold >= counter )
    CMPRET = arrayfire::ge(&threshold, &counter, false);
    //Lookup  1 >= dir_line  >= 0
    idx_intersect = arrayfire::locate(&CMPRET);

    //counter = arrayfire::lookup(&counter, &idx_intersect, 0);
    *start_line = arrayfire::lookup(start_line, &idx_intersect, 0);
    *dir_line = arrayfire::lookup(dir_line, &idx_intersect, 0);
    *input_idx = arrayfire::lookup(input_idx, &idx_intersect, 0);
    *hidden_idx = arrayfire::lookup(hidden_idx, &idx_intersect, 0);
}

pub fn line_line_intersect_2D(
    start_lines1: &arrayfire::Array<f32>,
    end_lines1: &arrayfire::Array<f32>,

    start_lines2: &arrayfire::Array<f32>,
    end_lines2: &arrayfire::Array<f32>,

    intersect: &mut arrayfire::Array<bool>,
) {
    let lines1_num = start_lines1.dims()[0];

    let lines2_num = start_lines2.dims()[0];

    let mut X1 = arrayfire::col(start_lines1, 0);
    let mut Y1 = arrayfire::col(start_lines1, 1);

    let mut X2 = arrayfire::col(end_lines1, 0);
    let mut Y2 = arrayfire::col(end_lines1, 1);

    let mut X3 = arrayfire::col(start_lines2, 0);
    X3 = arrayfire::reorder_v2(&X3, 2, 1, Some(vec![0]));

    let mut Y3 = arrayfire::col(start_lines2, 1);
    Y3 = arrayfire::reorder_v2(&Y3, 2, 1, Some(vec![0]));

    let mut X4 = arrayfire::col(end_lines2, 0);
    X4 = arrayfire::reorder_v2(&X4, 2, 1, Some(vec![0]));

    let mut Y4 = arrayfire::col(end_lines2, 1);
    Y4 = arrayfire::reorder_v2(&Y4, 2, 1, Some(vec![0]));

    let mut X1SUBX2 = arrayfire::sub(&X1, &X2, true);

    let mut Y4SUBY3 = arrayfire::sub(&Y4, &Y3, true);

    let mut Y1SUBY2 = arrayfire::sub(&Y1, &Y2, true);

    let mut X4SUBX3 = arrayfire::sub(&X4, &X3, true);

    let mut D = arrayfire::mul(&X1SUBX2, &Y4SUBY3, true) - arrayfire::mul(&Y1SUBY2, &X4SUBX3, true);

    drop(X2);
    drop(X4);

    drop(Y2);
    drop(Y4);

    let absD = arrayfire::abs(&D);

    //  ( absD >= epsilon )
    *intersect = arrayfire::ge(&absD, &epsilon, false);

    //Lookup  IDX (nabla >= 0 )
    let idx_intersect = arrayfire::locate(intersect);

    //if empty quit
    if (idx_intersect.dims()[0] == 0) {
        return;
    }

    D = arrayfire::flat(&D);

    D = arrayfire::lookup(&D, &idx_intersect, 0);

    //Get IDX 1
    let temp_idx1 = arrayfire::modulo(&idx_intersect, &lines1_num, false);

    //Lookup   Lines 1
    X1 = arrayfire::lookup(&X1, &temp_idx1, 0);
    Y1 = arrayfire::lookup(&Y1, &temp_idx1, 0);

    //X2 = arrayfire::lookup(&X2, &temp_idx1, 0);
    //Y2 = arrayfire::lookup(&Y2, &temp_idx1, 0);

    X1SUBX2 = arrayfire::lookup(&X1SUBX2, &temp_idx1, 0);
    Y1SUBY2 = arrayfire::lookup(&Y1SUBY2, &temp_idx1, 0);

    //Transpose 2
    X3 = arrayfire::reorder_v2(&X3, 2, 1, Some(vec![0]));
    Y3 = arrayfire::reorder_v2(&Y3, 2, 1, Some(vec![0]));

    //X4 = arrayfire::reorder_v2(&X4, 2, 1, Some(vec![0]));
    //Y4 = arrayfire::reorder_v2(&Y4, 2, 1, Some(vec![0]));

    X4SUBX3 = arrayfire::reorder_v2(&X4SUBX3, 2, 1, Some(vec![0]));
    Y4SUBY3 = arrayfire::reorder_v2(&Y4SUBY3, 2, 1, Some(vec![0]));

    //Get IDX 2
    let temp_idx2 = arrayfire::div(&idx_intersect, &lines1_num, false);

    //Lookup   Lines 2
    X3 = arrayfire::lookup(&X3, &temp_idx2, 0);
    Y3 = arrayfire::lookup(&Y3, &temp_idx2, 0);

    //X4 = arrayfire::lookup(&X4, &temp_idx2, 0);
    //Y4 = arrayfire::lookup(&Y4, &temp_idx2, 0);

    X4SUBX3 = arrayfire::lookup(&X4SUBX3, &temp_idx2, 0);
    Y4SUBY3 = arrayfire::lookup(&Y4SUBY3, &temp_idx2, 0);

    let X1SUBX3 = arrayfire::sub(&X1, &X3, false);

    //let X3SUBX4 =  -X4SUBX3.clone();

    let Y1SUBY3 = arrayfire::sub(&Y1, &Y3, false);

    drop(X1);
    drop(X3);

    drop(Y1);
    drop(Y3);

    let TX = (arrayfire::mul(&Y4SUBY3, &X1SUBX3, false)
        - arrayfire::mul(&X4SUBX3, &Y1SUBY3, false))
        / D.clone();

    //let Y2SUBY1 = -Y1SUBY2.clone();

    let TY = (-arrayfire::mul(&Y1SUBY2, &X1SUBX3, false)
        + arrayfire::mul(&X1SUBX2, &Y1SUBY3, false))
        / D;

    drop(Y4SUBY3);
    drop(X1SUBX3);
    drop(X4SUBX3);
    drop(Y1SUBY3);

    drop(Y1SUBY2);
    drop(X1SUBX2);

    //  (TX > epsilon )
    let mut CMPRET = arrayfire::gt(&TX, &epsilon, false);

    //  (oneminuseps > TX )
    let CMP1 = arrayfire::gt(&oneminuseps, &TX, false);
    CMPRET = arrayfire::and(&CMPRET, &CMP1, false);
    drop(CMP1);

    //  (TY > epsilon )
    let CMP2 = arrayfire::gt(&TY, &epsilon, false);
    CMPRET = arrayfire::and(&CMPRET, &CMP2, false);
    drop(CMP2);

    //  (oneminuseps > TY )
    let CMP3 = arrayfire::gt(&oneminuseps, &TY, false);
    CMPRET = arrayfire::and(&CMPRET, &CMP3, false);
    drop(CMP3);

    let mut newidx = arrayfire::locate(&CMPRET);
    drop(CMPRET);

    //Lookup
    newidx = arrayfire::lookup(&idx_intersect, &newidx, 0);

    let intersect_dims = arrayfire::Dim4::new(&[lines1_num * lines2_num, 1, 1, 1]);
    *intersect = arrayfire::constant::<bool>(false, intersect_dims);

    let inarr = arrayfire::constant::<bool>(true, newidx.dims());

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&newidx, 0, None);
    arrayfire::assign_gen(intersect, &idxrs, &inarr);

    let intersect_dims = arrayfire::Dim4::new(&[lines1_num, 1, lines2_num, 1]);
    *intersect = arrayfire::moddims(intersect, intersect_dims);
}

pub fn line_line_intersect(
    start_lines1: &arrayfire::Array<f32>,
    end_lines1: &arrayfire::Array<f32>,

    start_lines2: &arrayfire::Array<f32>,
    end_lines2: &arrayfire::Array<f32>,

    intersect: &mut arrayfire::Array<bool>,
) {
    let space_dims = start_lines1.dims()[1];

    if space_dims == 2 {
        line_line_intersect_2D(
            start_lines1,
            end_lines1,
            start_lines2,
            end_lines2,
            intersect,
        );
    } else {
    }
}

pub fn line_line_intersect_batch(
    batch_size: u64,
    start_lines1: &arrayfire::Array<f32>,
    end_lines1: &arrayfire::Array<f32>,

    start_lines2: &arrayfire::Array<f32>,
    end_lines2: &arrayfire::Array<f32>,

    intersect: &mut arrayfire::Array<bool>,
) {
    let mut i: u64 = 0;
    let mut startseq: u64 = 0;
    let mut endseq: u64 = 0;
    let total_size = start_lines2.dims()[0];

    startseq = i;
    endseq = i + batch_size - 1;
    if (endseq >= (total_size - 1)) {
        endseq = total_size - 1;
    }

    let seqs = &[
        arrayfire::Seq::new(startseq as f32, endseq as f32, 1.0 as f32),
        arrayfire::Seq::default(),
    ];
    let input_start_lines2 = arrayfire::index(start_lines2, seqs);

    let seqs2 = &[
        arrayfire::Seq::new(startseq as f32, endseq as f32, 1.0 as f32),
        arrayfire::Seq::default(),
    ];
    let input_end_lines2 = arrayfire::index(end_lines2, seqs2);

    line_line_intersect(
        start_lines1,
        end_lines1,
        &input_start_lines2,
        &input_end_lines2,
        intersect,
    );

    i = i + batch_size;

    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let mut intersect_temp = arrayfire::constant::<bool>(false, single_dims);

    while i < total_size {
        startseq = i;
        endseq = i + batch_size - 1;
        if (endseq >= (total_size - 1)) {
            endseq = total_size - 1;
        }

        let seqs = &[
            arrayfire::Seq::new(startseq as f32, endseq as f32, 1.0 as f32),
            arrayfire::Seq::default(),
        ];
        let input_start_lines2 = arrayfire::index(start_lines2, seqs);

        let seqs2 = &[
            arrayfire::Seq::new(startseq as f32, endseq as f32, 1.0 as f32),
            arrayfire::Seq::default(),
        ];
        let input_end_lines2 = arrayfire::index(end_lines2, seqs2);

        line_line_intersect(
            start_lines1,
            end_lines1,
            &input_start_lines2,
            &input_end_lines2,
            &mut intersect_temp,
        );

        *intersect = arrayfire::join(2, intersect, &intersect_temp);

        i = i + batch_size;
    }
}

/*
Raytracing algorithm 3 for creating neural connections. Connects all neurons within minibatches/groups of neurons

Inputs
raytrace_options:    Raytracing options
netdata:             Network metadata
glia_pos_total:      The positions of all glial cells
input_pos_total:     Selected neurons positions as source for the rays
input_idx_total:     Selected neurons positions as source for the rays
hidden_pos_total:    Selected neurons positions as targets for the rays
hidden_idx_total:    Selected neurons positions as targets for the rays


Outputs:
WRowIdxCOO:     Row vector in the COO sparse matrix
WColIdx:        Column vector in the COO sparse matrix

*/

pub fn RT3_distance_limited_directly_connected(
    raytrace_options: &raytrace_option_type,

    netdata: &network_metadata_type,
    glia_pos_total: &arrayfire::Array<f32>,

    input_pos_total: &arrayfire::Array<f32>,
    input_idx_total: &arrayfire::Array<i32>,

    hidden_pos_total: &arrayfire::Array<f32>,
    hidden_idx_total: &arrayfire::Array<i32>,

    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = input_pos_total.dims()[0];
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f32 = netdata.time_step.clone();
    let nratio: f32 = netdata.nratio.clone();
    let neuron_std: f32 = netdata.neuron_std.clone();
    let sphere_rad: f32 = netdata.sphere_rad.clone();
    let neuron_rad: f32 = netdata.neuron_rad.clone();
    let con_rad: f32 = netdata.con_rad.clone();
    let init_prob: f32 = netdata.init_prob.clone();

    let input_connection_num: u64 = raytrace_options.input_connection_num.clone();
    let max_rounds: u64 = raytrace_options.max_rounds.clone();
    let ray_glia_intersect: bool = raytrace_options.ray_glia_intersect.clone();
    let ray_neuron_intersect: bool = raytrace_options.ray_neuron_intersect.clone();

    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut gidxOld = arrayfire::constant::<u64>(0, single_dims);

    let mut gidxOld_cpu: Vec<u64> = Vec::new();

    let WColIdxelements = WColIdx.elements();

    //let mut WValues_cpu = Vec::new();
    let mut WRowIdxCOO_cpu = Vec::new();
    let mut WColIdx_cpu = Vec::new();

    //Insert previous COO Matrix values
    if WColIdxelements > 1 {
        //TO CPU

        WRowIdxCOO_cpu = vec![i32::default(); WRowIdxCOO.elements()];
        WRowIdxCOO.host(&mut WRowIdxCOO_cpu);

        WColIdx_cpu = vec![i32::default(); WColIdx.elements()];
        WColIdx.host(&mut WColIdx_cpu);

        //Compute global index
        gidxOld = get_global_weight_idx(neuron_size, &WRowIdxCOO, &WColIdx);

        //TO CPU
        gidxOld_cpu = vec![u64::default(); gidxOld.elements()];
        gidxOld.host(&mut gidxOld_cpu);
        drop(gidxOld);

        *WRowIdxCOO = arrayfire::constant::<i32>(0, single_dims);
        *WColIdx = arrayfire::constant::<i32>(0, single_dims);
    }

    let mut hidden_idx = arrayfire::constant::<i32>(0, single_dims);

    let mut tiled_input_idx = arrayfire::constant::<i32>(0, single_dims);

    let mut tiled_hidden_idx = arrayfire::constant::<i32>(0, single_dims);

    //Get input and hidden positions

    let mut hidden_pos = arrayfire::constant::<f32>(0.0, single_dims);

    let mut circle_radius = arrayfire::constant::<f32>(
        neuron_rad,
        arrayfire::Dim4::new(&[hidden_pos.dims()[0], 1, 1, 1]),
    );

    let mut start_line = arrayfire::constant::<f32>(0.0, single_dims);
    let mut dir_line = arrayfire::constant::<f32>(0.0, single_dims);

    let mut raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / start_line.dims()[0]) as u64);

    let mut hidden_size_u32 = hidden_pos_total.dims()[0] as u32;
    let mut hidden_size = hidden_pos_total.dims()[0];

    //Store COO Matrix values

    let mut join_all = Vec::new();

    let mut gidx1 = arrayfire::constant::<u64>(0, single_dims);

    let mut gidx1_cpu: Vec<u64> = Vec::new();

    let mut prev_con_num = 0;

    let mut input_idx_size = 0;

    let mut rng = rand::thread_rng();
    let rand_vec: Vec<u64> = (0..input_size).collect();
    let mut select_input_idx: u64 = 0;

    let mut input_pos = arrayfire::constant::<f32>(0.0, single_dims);
    let mut input_idx = arrayfire::constant::<i32>(0, single_dims);

    let mut glia_pos = arrayfire::constant::<f32>(0.0, single_dims);
    let mut glia_idx = arrayfire::constant::<i32>(0, single_dims);

    let mut same_counter: u64 = 0;
    let mut pivot_pos = vec![-sphere_rad * 0.7f32; space_dims as usize];
    let pivot_rad = 4.0f32 * con_rad;
    let mut nonoverlapping = true;

    for vv in 0..max_rounds {
        select_input_idx = rand_vec.choose(&mut rng).unwrap().clone();
        let mut target_input = arrayfire::row(input_pos_total, select_input_idx as i64);

        input_pos = input_pos_total.clone();
        input_idx = input_idx_total.clone();

        filter_rays(
            2.0f32 * con_rad,
            &target_input,
            &mut input_pos,
            &mut input_idx,
        );

        if input_idx.dims()[0] == 0 {
            continue;
        }

        input_idx_size = input_idx.dims()[0];

        hidden_pos = hidden_pos_total.clone();
        hidden_idx = hidden_idx_total.clone();

        filter_rays(con_rad, &target_input, &mut hidden_pos, &mut hidden_idx);

        hidden_size = hidden_idx.dims()[0];
        hidden_size_u32 = hidden_idx.dims()[0] as u32;

        if hidden_size == 0 {
            continue;
        }

        //Generate rays starting from input neurons
        start_line = arrayfire::constant::<f32>(0.0, single_dims);
        dir_line = arrayfire::constant::<f32>(0.0, single_dims);

        let tile_dims = arrayfire::Dim4::new(&[hidden_size, 1, 1, 1]);

        tiled_input_idx = arrayfire::tile(&input_idx, tile_dims);
        drop(input_idx);

        tiled_hidden_idx = hidden_idx.clone();
        drop(hidden_idx);

        tileDown_i32(input_idx_size, &mut tiled_hidden_idx);

        //println!("z1");
        //println!("input_pos.dims()[0] {}",input_pos.dims()[0]);
        //println!("input_pos.dims()[1] {}",input_pos.dims()[1]);
        //println!("hidden_pos.dims()[0] {}",hidden_pos.dims()[0]);
        //println!("hidden_pos.dims()[1] {}",hidden_pos.dims()[1]);
        //println!("con_rad {}", con_rad);

        rays_from_neuronsA_to_neuronsB(
            con_rad,
            &input_pos,
            &hidden_pos,
            &mut start_line,
            &mut dir_line,
            &mut tiled_input_idx,
            &mut tiled_hidden_idx,
        );
        drop(input_pos);

        //println!("z1");
        //println!("start_line.dims()[0] {}",start_line.dims()[0]);
        //println!("start_line.dims()[1] {}",start_line.dims()[1]);
        //println!("dir_line.dims()[0] {}",dir_line.dims()[0]);
        //println!("dir_line.dims()[1] {}",dir_line.dims()[1]);
        //println!("tiled_input_idx.dims()[0] {}",tiled_input_idx.dims()[0]);
        //println!("tiled_input_idx.dims()[1] {}",tiled_input_idx.dims()[1]);
        //println!("tiled_hidden_idx.dims()[0] {}",tiled_hidden_idx.dims()[0]);
        //println!("tiled_hidden_idx.dims()[1] {}",tiled_hidden_idx.dims()[1]);

        if start_line.dims()[0] == 0 {
            continue;
        }

        raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / start_line.dims()[0]) as u64);

        circle_radius = arrayfire::constant::<f32>(
            neuron_rad,
            arrayfire::Dim4::new(&[hidden_pos.dims()[0], 1, 1, 1]),
        );

        if ray_neuron_intersect && (hidden_size > 1) {
            line_sphere_intersect_batchV2(
                raytrace_batch_size,
                2,
                &hidden_pos,
                &circle_radius,
                &mut start_line,
                &mut dir_line,
                &mut tiled_input_idx,
                &mut tiled_hidden_idx,
            );
        }
        drop(hidden_pos);

        if tiled_input_idx.dims()[0] == 0 {
            continue;
        }

        //println!("a1");
        //println!("intersect.dims()[0] {}",intersect.dims()[0]);
        //println!("intersect.dims()[1] {}",intersect.dims()[1]);
        //println!("intersect.dims()[2] {}",intersect.dims()[2]);

        //println!("a2");
        //println!("intersect.dims()[0] {}",intersect.dims()[0]);
        //println!("intersect.dims()[1] {}",intersect.dims()[1]);
        //println!("intersect.dims()[2] {}",intersect.dims()[2]);

        //println!("input_size {}",input_size);

        glia_pos = glia_pos_total.clone();
        glia_idx = arrayfire::constant::<i32>(0, glia_pos.dims());

        filter_rays(con_rad, &target_input, &mut glia_pos, &mut glia_idx);
        drop(glia_idx);

        raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / start_line.dims()[0]) as u64);

        circle_radius = arrayfire::constant::<f32>(
            neuron_rad,
            arrayfire::Dim4::new(&[glia_pos.dims()[0], 1, 1, 1]),
        );

        if ray_glia_intersect && (glia_pos.dims()[0] > 1) {
            line_sphere_intersect_batchV2(
                raytrace_batch_size,
                0,
                &glia_pos,
                &circle_radius,
                &mut start_line,
                &mut dir_line,
                &mut tiled_input_idx,
                &mut tiled_hidden_idx,
            );
        }
        drop(glia_pos);

        if tiled_input_idx.dims()[0] == 0 {
            continue;
        }

        //Compute global index
        gidx1 = get_global_weight_idx(neuron_size, &tiled_hidden_idx, &tiled_input_idx);

        let mut gidx1_cpu = vec![u64::default(); gidx1.elements()];
        gidx1.host(&mut gidx1_cpu);
        drop(gidx1);

        let mut tiled_hidden_idx_cpu = vec![i32::default(); tiled_hidden_idx.elements()];
        tiled_hidden_idx.host(&mut tiled_hidden_idx_cpu);
        drop(tiled_hidden_idx);

        let mut tiled_input_idx_cpu = vec![i32::default(); tiled_input_idx.elements()];
        tiled_input_idx.host(&mut tiled_input_idx_cpu);
        drop(tiled_input_idx);

        //println!("join_WColIdx.len() {}", join_WColIdx.len());
        //println!("join_WColIdx.keys().len() {}", join_WColIdx.keys().len());
        //println!("gidx1_cpu.len() {}", gidx1_cpu.len());

        //Save new neural connections to COO matrix hashmap
        for qq in 0..gidx1_cpu.len() {
            join_all.push((
                gidx1_cpu[qq].clone(),
                tiled_input_idx_cpu[qq].clone(),
                tiled_hidden_idx_cpu[qq].clone(),
            ));
        }

        //println!("join_WColIdx.len() {}", join_WColIdx.len());
        if ((join_all.len() as u64) > (input_connection_num)) {
            join_all.par_sort_unstable_by_key(|pair| pair.0);
            join_all.dedup_by_key(|pair| pair.0);

            if ((join_all.len() as u64) > (input_connection_num)) {
                break;
            }
        }

        if ((join_all.len() as u64) > prev_con_num) {
            prev_con_num = join_all.len() as u64;
            same_counter = 0;
        } else {
            same_counter = same_counter + 1;
        }

        if same_counter > 5 {
            break;
        }
    }

    drop(gidx1_cpu);
    //Insert previous COO Matrix values
    if WColIdxelements > 1 {
        //Insert old value
        for qq in 0..gidxOld_cpu.len() {
            join_all.push((
                gidxOld_cpu[qq].clone(),
                WColIdx_cpu[qq].clone(),
                WRowIdxCOO_cpu[qq].clone(),
            ));
        }
        drop(gidxOld_cpu);
        //println!("join_WValues.len() {}", join_WValues.len());
    }

    //Sort global index
    join_all.par_sort_unstable_by_key(|pair| pair.0);
    join_all.dedup_by_key(|pair| pair.0);

    let (WColIdx_cpu, WRowIdxCOO_cpu): (Vec<_>, Vec<_>) =
        join_all.par_iter().cloned().map(|(_, b, c)| (b, c)).unzip();

    //Convert cpu vector to gpu array
    *WColIdx = arrayfire::Array::new(
        &WColIdx_cpu,
        arrayfire::Dim4::new(&[WColIdx_cpu.len() as u64, 1, 1, 1]),
    );
    *WRowIdxCOO = arrayfire::Array::new(
        &WRowIdxCOO_cpu,
        arrayfire::Dim4::new(&[WColIdx_cpu.len() as u64, 1, 1, 1]),
    );
}

pub fn raytrace_hidden_layers(
    netdata: &network_metadata_type,
    neuron_pos: &arrayfire::Array<f32>,
    neuron_idx: &arrayfire::Array<i32>,

    WValues: &mut arrayfire::Array<f32>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f32 = netdata.time_step.clone();
    let nratio: f32 = netdata.nratio.clone();
    let neuron_std: f32 = netdata.neuron_std.clone();
    let sphere_rad: f32 = netdata.sphere_rad.clone();
    let neuron_rad: f32 = netdata.neuron_rad.clone();
    let con_rad: f32 = netdata.con_rad.clone();
    let init_prob: f32 = netdata.init_prob.clone();

    let neuron_dims = neuron_pos.dims();
    let neuron_num = neuron_dims[0];

    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let colseq = arrayfire::Seq::new(0.0, (space_dims - 1) as f32, 1.0);

    let mut temparr =
        arrayfire::constant::<f32>(0.0, arrayfire::Dim4::new(&[neuron_size, space_dims, 1, 1]));

    let idx = neuron_idx.clone();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idx, 0, None);
    idxrs.set_index(&colseq, 1, Some(false));
    arrayfire::assign_gen(&mut temparr, &idxrs, neuron_pos);

    let active_size = neuron_idx.dims()[0];
    let active_size_u32 = active_size as u32;

    let circle_radius =
        arrayfire::constant::<f32>(neuron_rad, arrayfire::Dim4::new(&[active_size, 1, 1, 1]));

    let mut curidxsel = arrayfire::rows(
        neuron_idx,
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );
    let mut idxsel = curidxsel.clone();

    let mut cur_num = idxsel.dims()[0];
    let mut cur_num_u32 = idxsel.dims()[0] as u32;

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idxsel, 0, None);
    idxrs.set_index(&colseq, 1, Some(false));
    let mut cur_neuron_pos = arrayfire::index_gen(&temparr, idxrs);

    let mut start_line = arrayfire::constant::<f32>(0.0, single_dims);
    let mut dir_line = arrayfire::constant::<f32>(0.0, single_dims);

    generate_random_uniform_rays(
        &cur_neuron_pos,
        AVG_RAYTRACE_NUM,
        con_rad,
        &mut start_line,
        &mut dir_line,
    );

    let mut line_num = start_line.dims()[0] as u32;

    let mut raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / start_line.dims()[0]) as u64);

    let mut intersect = arrayfire::constant::<bool>(false, single_dims);

    line_sphere_intersect_batch(
        raytrace_batch_size,
        &start_line,
        &dir_line,
        neuron_pos,
        &circle_radius,
        &mut intersect,
    );

    intersect = arrayfire::reorder_v2(&intersect, 2, 1, Some(vec![0]));

    let mut idx_intersect = arrayfire::locate(&intersect);

    let mut div_idx = arrayfire::div(&idx_intersect, &active_size_u32, false);

    let mut mod_idx = arrayfire::modulo(&idx_intersect, &active_size_u32, false);

    let (key, value) = arrayfire::min_by_key(&div_idx, &mod_idx, 0);

    *WRowIdxCOO = arrayfire::modulo(&key, &cur_num_u32, false).cast::<i32>();

    *WColIdx = value.cast::<i32>();

    let mut global_idx = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);

    global_idx = arrayfire::set_unique(&global_idx, false);

    *WRowIdxCOO = arrayfire::div(&global_idx, &neuron_size, false).cast::<i32>();

    *WColIdx = arrayfire::modulo(&global_idx, &neuron_size, false).cast::<i32>();

    *WRowIdxCOO = arrayfire::lookup(&idxsel, WRowIdxCOO, 0);

    *WValues = neuron_std * arrayfire::randn::<f32>(WRowIdxCOO.dims());

    curidxsel = WColIdx.clone();

    curidxsel = find_unique(&curidxsel, neuron_size);

    let mut newWRowIdxCOO = arrayfire::constant::<i32>(0, single_dims);

    let mut newWColIdx = arrayfire::constant::<i32>(0, single_dims);

    let mut start_lines1 = arrayfire::constant::<f32>(0.0, single_dims);
    let mut end_lines1 = arrayfire::constant::<f32>(0.0, single_dims);

    let mut start_lines2 = arrayfire::constant::<f32>(0.0, single_dims);
    let mut end_lines2 = arrayfire::constant::<f32>(0.0, single_dims);

    for i in 1..proc_num {
        idxsel = curidxsel.clone();

        cur_num = idxsel.dims()[0];
        cur_num_u32 = idxsel.dims()[0] as u32;

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&idxsel, 0, None);
        idxrs.set_index(&colseq, 1, Some(false));
        cur_neuron_pos = arrayfire::index_gen(&temparr, idxrs);

        start_line = arrayfire::constant::<f32>(0.0, single_dims);
        dir_line = arrayfire::constant::<f32>(0.0, single_dims);

        generate_random_uniform_rays(
            &cur_neuron_pos,
            AVG_RAYTRACE_NUM,
            con_rad,
            &mut start_line,
            &mut dir_line,
        );

        line_num = start_line.dims()[0] as u32;

        raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / start_line.dims()[0]) as u64);

        intersect = arrayfire::constant::<bool>(false, single_dims);

        line_sphere_intersect_batch(
            raytrace_batch_size,
            &start_line,
            &dir_line,
            neuron_pos,
            &circle_radius,
            &mut intersect,
        );

        intersect = arrayfire::reorder_v2(&intersect, 2, 1, Some(vec![0]));

        idx_intersect = arrayfire::locate(&intersect);

        div_idx = arrayfire::div(&idx_intersect, &active_size_u32, false);

        mod_idx = arrayfire::modulo(&idx_intersect, &active_size_u32, false);

        let (key, value) = arrayfire::min_by_key(&div_idx, &mod_idx, 0);

        newWRowIdxCOO = arrayfire::modulo(&key, &cur_num_u32, false).cast::<i32>();

        newWColIdx = value.cast::<i32>();

        newWRowIdxCOO = arrayfire::lookup(&idxsel, &newWRowIdxCOO, 0);

        *WRowIdxCOO = arrayfire::join(0, &newWRowIdxCOO, WRowIdxCOO);

        *WColIdx = arrayfire::join(0, &newWColIdx, WColIdx);

        global_idx = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);

        global_idx = arrayfire::set_unique(&global_idx, false);

        *WRowIdxCOO = arrayfire::div(&global_idx, &neuron_size, false).cast::<i32>();

        *WColIdx = arrayfire::modulo(&global_idx, &neuron_size, false).cast::<i32>();

        *WValues = neuron_std * arrayfire::randn::<f32>(WRowIdxCOO.dims());

        curidxsel = WColIdx.clone();

        curidxsel = find_unique(&curidxsel, neuron_size);
    }
}

pub fn raytrace_hidden_layers2(
    con_num: u64,

    netdata: &network_metadata_type,
    neuron_pos: &arrayfire::Array<f32>,
    neuron_idx: &arrayfire::Array<i32>,

    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f32 = netdata.time_step.clone();
    let nratio: f32 = netdata.nratio.clone();
    let neuron_std: f32 = netdata.neuron_std.clone();
    let sphere_rad: f32 = netdata.sphere_rad.clone();
    let neuron_rad: f32 = netdata.neuron_rad.clone();
    let con_rad: f32 = netdata.con_rad.clone();
    let init_prob: f32 = netdata.init_prob.clone();

    let neuron_dims = neuron_pos.dims();
    let neuron_num = neuron_dims[0];

    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let colseq = arrayfire::Seq::new(0.0, (space_dims - 1) as f32, 1.0);

    let mut temparr =
        arrayfire::constant::<f32>(0.0, arrayfire::Dim4::new(&[neuron_size, space_dims, 1, 1]));

    let idx = neuron_idx.clone();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idx, 0, None);
    idxrs.set_index(&colseq, 1, Some(false));
    arrayfire::assign_gen(&mut temparr, &idxrs, neuron_pos);

    let active_size = neuron_idx.dims()[0];
    let active_size_u32 = active_size as u32;

    let circle_radius =
        arrayfire::constant::<f32>(neuron_rad, arrayfire::Dim4::new(&[active_size, 1, 1, 1]));

    let mut curidxsel = arrayfire::rows(
        neuron_idx,
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );
    let mut idxsel = curidxsel.clone();

    let mut cur_num = idxsel.dims()[0];
    let mut cur_num_u32 = idxsel.dims()[0] as u32;

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idxsel, 0, None);
    idxrs.set_index(&colseq, 1, Some(false));
    let mut cur_neuron_pos = arrayfire::index_gen(&temparr, idxrs);

    let mut start_line = arrayfire::constant::<f32>(0.0, single_dims);
    let mut dir_line = arrayfire::constant::<f32>(0.0, single_dims);

    generate_random_uniform_rays(
        &cur_neuron_pos,
        AVG_RAYTRACE_NUM,
        con_rad,
        &mut start_line,
        &mut dir_line,
    );

    let mut line_num = start_line.dims()[0] as u32;

    let mut raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / start_line.dims()[0]) as u64);

    let mut intersect = arrayfire::constant::<bool>(false, single_dims);

    line_sphere_intersect_batch(
        raytrace_batch_size,
        &start_line,
        &dir_line,
        neuron_pos,
        &circle_radius,
        &mut intersect,
    );

    intersect = arrayfire::reorder_v2(&intersect, 2, 1, Some(vec![0]));

    let mut idx_intersect = arrayfire::locate(&intersect);

    let mut div_idx = arrayfire::div(&idx_intersect, &active_size_u32, false);

    let mut mod_idx = arrayfire::modulo(&idx_intersect, &active_size_u32, false);

    let (key, value) = arrayfire::min_by_key(&div_idx, &mod_idx, 0);

    *WRowIdxCOO = arrayfire::modulo(&key, &cur_num_u32, false).cast::<i32>();

    *WColIdx = value.cast::<i32>();

    let mut global_idx = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);

    global_idx = arrayfire::set_unique(&global_idx, false);

    *WRowIdxCOO = arrayfire::div(&global_idx, &neuron_size, false).cast::<i32>();

    *WColIdx = arrayfire::modulo(&global_idx, &neuron_size, false).cast::<i32>();

    *WRowIdxCOO = arrayfire::lookup(&idxsel, WRowIdxCOO, 0);

    curidxsel = WColIdx.clone();

    curidxsel = find_unique(&curidxsel, neuron_size);

    let mut newWRowIdxCOO = arrayfire::constant::<i32>(0, single_dims);

    let mut newWColIdx = arrayfire::constant::<i32>(0, single_dims);

    let mut start_lines1 = arrayfire::constant::<f32>(0.0, single_dims);
    let mut end_lines1 = arrayfire::constant::<f32>(0.0, single_dims);

    let mut start_lines2 = arrayfire::constant::<f32>(0.0, single_dims);
    let mut end_lines2 = arrayfire::constant::<f32>(0.0, single_dims);

    loop {
        idxsel = curidxsel.clone();

        cur_num = idxsel.dims()[0];
        cur_num_u32 = idxsel.dims()[0] as u32;

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&idxsel, 0, None);
        idxrs.set_index(&colseq, 1, Some(false));
        cur_neuron_pos = arrayfire::index_gen(&temparr, idxrs);

        start_line = arrayfire::constant::<f32>(0.0, single_dims);
        dir_line = arrayfire::constant::<f32>(0.0, single_dims);

        generate_random_uniform_rays(
            &cur_neuron_pos,
            AVG_RAYTRACE_NUM,
            con_rad,
            &mut start_line,
            &mut dir_line,
        );

        line_num = start_line.dims()[0] as u32;

        raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / start_line.dims()[0]) as u64);

        intersect = arrayfire::constant::<bool>(false, single_dims);

        line_sphere_intersect_batch(
            raytrace_batch_size,
            &start_line,
            &dir_line,
            neuron_pos,
            &circle_radius,
            &mut intersect,
        );

        intersect = arrayfire::reorder_v2(&intersect, 2, 1, Some(vec![0]));

        idx_intersect = arrayfire::locate(&intersect);

        div_idx = arrayfire::div(&idx_intersect, &active_size_u32, false);

        mod_idx = arrayfire::modulo(&idx_intersect, &active_size_u32, false);

        let (key, value) = arrayfire::min_by_key(&div_idx, &mod_idx, 0);

        newWRowIdxCOO = arrayfire::modulo(&key, &cur_num_u32, false).cast::<i32>();

        newWColIdx = value.cast::<i32>();

        newWRowIdxCOO = arrayfire::lookup(&idxsel, &newWRowIdxCOO, 0);

        *WRowIdxCOO = arrayfire::join(0, &newWRowIdxCOO, WRowIdxCOO);

        *WColIdx = arrayfire::join(0, &newWColIdx, WColIdx);

        global_idx = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);

        global_idx = arrayfire::set_unique(&global_idx, false);

        *WRowIdxCOO = arrayfire::div(&global_idx, &neuron_size, false).cast::<i32>();

        *WColIdx = arrayfire::modulo(&global_idx, &neuron_size, false).cast::<i32>();

        if WRowIdxCOO.dims()[0] > con_num {
            break;
        }

        curidxsel = WColIdx.clone();

        curidxsel = find_unique(&curidxsel, neuron_size);
    }
}

pub fn delete_intersecting_lines(
    netdata: &network_metadata_type,
    neuron_pos: &arrayfire::Array<f32>,
    neuron_idx: &arrayfire::Array<i32>,

    WValues: &mut arrayfire::Array<f32>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f32 = netdata.time_step.clone();
    let nratio: f32 = netdata.nratio.clone();
    let neuron_std: f32 = netdata.neuron_std.clone();
    let sphere_rad: f32 = netdata.sphere_rad.clone();
    let neuron_rad: f32 = netdata.neuron_rad.clone();
    let con_rad: f32 = netdata.con_rad.clone();
    let init_prob: f32 = netdata.init_prob.clone();

    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let colseq = arrayfire::Seq::new(0.0, (space_dims - 1) as f32, 1.0);

    let mut temparr =
        arrayfire::constant::<f32>(0.0, arrayfire::Dim4::new(&[neuron_size, space_dims, 1, 1]));

    let idx = neuron_idx.clone();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idx, 0, None);
    idxrs.set_index(&colseq, 1, Some(false));
    arrayfire::assign_gen(&mut temparr, &idxrs, neuron_pos);

    let mut global_idx = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);

    global_idx = arrayfire::set_unique(&global_idx, false);

    *WRowIdxCOO = arrayfire::div(&global_idx, &neuron_size, false).cast::<i32>();

    *WColIdx = arrayfire::modulo(&global_idx, &neuron_size, false).cast::<i32>();

    *WValues = arrayfire::constant::<f32>(0.0, single_dims);

    //Loop to delete connections
    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(WColIdx, 0, None);
    idxrs.set_index(&colseq, 1, Some(false));
    let mut start_lines1 = arrayfire::index_gen(&temparr, idxrs);

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(WRowIdxCOO, 0, None);
    idxrs.set_index(&colseq, 1, Some(false));
    let mut end_lines1 = arrayfire::index_gen(&temparr, idxrs);

    let mut line_num3 = start_lines1.dims()[0] as u32;

    let mut raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / start_lines1.dims()[0]) as u64);

    let mut intersect = arrayfire::constant::<bool>(false, single_dims);

    line_line_intersect_batch(
        raytrace_batch_size,
        &start_lines1,
        &end_lines1,
        &start_lines1,
        &end_lines1,
        &mut intersect,
    );

    let mut idx_intersect = arrayfire::locate(&intersect);

    let mut div_idx = arrayfire::constant::<u32>(0, single_dims);

    let mut ones = arrayfire::constant::<u32>(0, single_dims);

    let mut top_keys = arrayfire::constant::<u32>(0, single_dims);

    let mut table = arrayfire::constant::<bool>(false, single_dims);

    let mut inarr = arrayfire::constant::<bool>(false, single_dims);

    let mut table_idx = arrayfire::constant::<u32>(0, single_dims);

    while idx_intersect.dims()[0] > 0 {
        line_num3 = start_lines1.dims()[0] as u32;

        div_idx = arrayfire::div(&idx_intersect, &line_num3, false);

        drop(idx_intersect);

        ones = arrayfire::constant::<u32>(1, div_idx.dims());

        let (key, value) = arrayfire::sum_by_key(&div_idx, &ones, 0);

        drop(div_idx);
        drop(ones);

        let keydim0 = key.dims()[0];

        if keydim0 <= 1 {
            break;
        }

        let mut key_num = (keydim0 / 100) as u32;

        if key_num <= 1 {
            key_num = 1;
        }

        if key_num > 255 {
            key_num = 255;
        }

        let (_, value_idx) = arrayfire::topk(&value, key_num, 0, arrayfire::TopkFn::MAX);

        drop(value);

        top_keys = arrayfire::lookup(&key, &value_idx, 0);

        drop(key);

        table = arrayfire::constant::<bool>(true, WColIdx.dims());

        inarr = arrayfire::constant::<bool>(false, top_keys.dims());
        //let idxarr = arr.cast::<u32>();

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&top_keys, 0, None);
        arrayfire::assign_gen(&mut table, &idxrs, &inarr);

        drop(inarr);

        table_idx = arrayfire::locate(&table);

        if table_idx.dims()[0] <= 1 {
            break;
        }

        *WRowIdxCOO = arrayfire::lookup(WRowIdxCOO, &table_idx, 0);

        *WColIdx = arrayfire::lookup(WColIdx, &table_idx, 0);

        drop(table_idx);

        //New data
        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(WColIdx, 0, None);
        idxrs.set_index(&colseq, 1, Some(false));
        start_lines1 = arrayfire::index_gen(&temparr, idxrs);

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(WRowIdxCOO, 0, None);
        idxrs.set_index(&colseq, 1, Some(false));
        end_lines1 = arrayfire::index_gen(&temparr, idxrs);

        raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / start_lines1.dims()[0]) as u64);

        line_line_intersect_batch(
            raytrace_batch_size,
            &start_lines1,
            &end_lines1,
            &start_lines1,
            &end_lines1,
            &mut intersect,
        );

        //drop(start_lines1);
        //drop(end_lines1);

        idx_intersect = arrayfire::locate(&intersect);
    }

    *WValues = neuron_std * arrayfire::randn::<f32>(WRowIdxCOO.dims());
}

/*
Raytracing algorithm 1 for creating neural connections. Randomly generates rays of random directions with variable number of random rays

Inputs
ray_num:        Number of rays per neuron per iteration
con_num:        Target number of total connections
netdata:        Network metadata
neuron_pos:     Neuron positions
neuron_idx:     Indexes of neuron positions

Outputs:
WRowIdxCOO:     Row vector in the COO sparse matrix
WColIdx:        Column vector in the COO sparse matrix

*/

pub fn RT1_random_rays(
    ray_num: u64,
    con_num: u64,

    netdata: &network_metadata_type,
    neuron_pos: &arrayfire::Array<f32>,
    neuron_idx: &arrayfire::Array<i32>,

    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f32 = netdata.time_step.clone();
    let nratio: f32 = netdata.nratio.clone();
    let neuron_std: f32 = netdata.neuron_std.clone();
    let sphere_rad: f32 = netdata.sphere_rad.clone();
    let neuron_rad: f32 = netdata.neuron_rad.clone();
    let con_rad: f32 = netdata.con_rad.clone();
    let init_prob: f32 = netdata.init_prob.clone();

    let neuron_dims = neuron_pos.dims();
    let neuron_num = neuron_dims[0];

    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let colseq = arrayfire::Seq::new(0.0, (space_dims - 1) as f32, 1.0);

    let mut temparr =
        arrayfire::constant::<f32>(0.0, arrayfire::Dim4::new(&[neuron_size, space_dims, 1, 1]));

    let idx = neuron_idx.clone();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idx, 0, None);
    idxrs.set_index(&colseq, 1, Some(false));
    arrayfire::assign_gen(&mut temparr, &idxrs, neuron_pos);

    let active_size = neuron_idx.dims()[0];
    let active_size_u32 = active_size as u32;

    let circle_radius =
        arrayfire::constant::<f32>(neuron_rad, arrayfire::Dim4::new(&[active_size, 1, 1, 1]));

    let mut curidxsel = arrayfire::rows(
        neuron_idx,
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );
    let mut idxsel = curidxsel.clone();

    let mut cur_num = idxsel.dims()[0];
    let mut cur_num_u32 = idxsel.dims()[0] as u32;

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idxsel, 0, None);
    idxrs.set_index(&colseq, 1, Some(false));
    let mut cur_neuron_pos = arrayfire::index_gen(&temparr, idxrs);

    let mut start_line = arrayfire::constant::<f32>(0.0, single_dims);
    let mut dir_line = arrayfire::constant::<f32>(0.0, single_dims);

    generate_random_uniform_rays(
        &cur_neuron_pos,
        ray_num,
        con_rad,
        &mut start_line,
        &mut dir_line,
    );

    let randarr_dims = arrayfire::Dim4::new(&[dir_line.dims()[0], 1, 1, 1]);
    let randarr = arrayfire::randu::<f32>(randarr_dims);
    let (_, mut randidx) = arrayfire::sort_index(&randarr, 0, false);

    let mut raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / (2 * neuron_pos.dims()[0])) as u64);

    if randidx.dims()[0] > raytrace_batch_size {
        randidx = arrayfire::rows(&randidx, 0, (raytrace_batch_size - 1) as i64);
    }

    let mut idxrs1 = arrayfire::Indexer::default();
    let seq1 = arrayfire::Seq::new(0.0, (space_dims - 1) as f32, 1.0);
    idxrs1.set_index(&randidx, 0, None);
    idxrs1.set_index(&seq1, 1, Some(false));
    start_line = arrayfire::index_gen(&start_line, idxrs1);

    let mut idxrs1 = arrayfire::Indexer::default();
    let seq1 = arrayfire::Seq::new(0.0, (space_dims - 1) as f32, 1.0);
    idxrs1.set_index(&randidx, 0, None);
    idxrs1.set_index(&seq1, 1, Some(false));
    dir_line = arrayfire::index_gen(&dir_line, idxrs1);

    let mut line_num = start_line.dims()[0] as u32;

    let mut raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / (start_line.dims()[0])) as u64);

    let mut intersect = arrayfire::constant::<bool>(false, single_dims);

    line_sphere_intersect_batch(
        raytrace_batch_size,
        &start_line,
        &dir_line,
        neuron_pos,
        &circle_radius,
        &mut intersect,
    );

    intersect = arrayfire::reorder_v2(&intersect, 2, 1, Some(vec![0]));

    let mut idx_intersect = arrayfire::locate(&intersect);

    let mut div_idx = arrayfire::div(&idx_intersect, &active_size_u32, false);

    let mut mod_idx = arrayfire::modulo(&idx_intersect, &active_size_u32, false);

    let (key, value) = arrayfire::min_by_key(&div_idx, &mod_idx, 0);

    *WRowIdxCOO = arrayfire::modulo(&key, &cur_num_u32, false).cast::<i32>();

    *WColIdx = value.cast::<i32>();

    let mut global_idx = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);

    global_idx = arrayfire::set_unique(&global_idx, false);

    *WRowIdxCOO = arrayfire::div(&global_idx, &neuron_size, false).cast::<i32>();

    *WColIdx = arrayfire::modulo(&global_idx, &neuron_size, false).cast::<i32>();

    *WRowIdxCOO = arrayfire::lookup(&idxsel, WRowIdxCOO, 0);

    curidxsel = WColIdx.clone();

    curidxsel = find_unique(&curidxsel, neuron_size);

    let mut newWRowIdxCOO = arrayfire::constant::<i32>(0, single_dims);

    let mut newWColIdx = arrayfire::constant::<i32>(0, single_dims);

    let mut start_lines1 = arrayfire::constant::<f32>(0.0, single_dims);
    let mut end_lines1 = arrayfire::constant::<f32>(0.0, single_dims);

    let mut start_lines2 = arrayfire::constant::<f32>(0.0, single_dims);
    let mut end_lines2 = arrayfire::constant::<f32>(0.0, single_dims);

    loop {
        idxsel = curidxsel.clone();

        cur_num = idxsel.dims()[0];
        cur_num_u32 = idxsel.dims()[0] as u32;

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&idxsel, 0, None);
        idxrs.set_index(&colseq, 1, Some(false));
        cur_neuron_pos = arrayfire::index_gen(&temparr, idxrs);

        start_line = arrayfire::constant::<f32>(0.0, single_dims);
        dir_line = arrayfire::constant::<f32>(0.0, single_dims);

        generate_random_uniform_rays(
            &cur_neuron_pos,
            ray_num,
            con_rad,
            &mut start_line,
            &mut dir_line,
        );

        let randarr_dims = arrayfire::Dim4::new(&[dir_line.dims()[0], 1, 1, 1]);
        let randarr = arrayfire::randu::<f32>(randarr_dims);
        let (_, mut randidx) = arrayfire::sort_index(&randarr, 0, false);

        let mut raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / (2 * neuron_pos.dims()[0])) as u64);

        if randidx.dims()[0] > raytrace_batch_size {
            randidx = arrayfire::rows(&randidx, 0, (raytrace_batch_size - 1) as i64);
        }

        let mut idxrs1 = arrayfire::Indexer::default();
        let seq1 = arrayfire::Seq::new(0.0, (space_dims - 1) as f32, 1.0);
        idxrs1.set_index(&randidx, 0, None);
        idxrs1.set_index(&seq1, 1, Some(false));
        start_line = arrayfire::index_gen(&start_line, idxrs1);

        let mut idxrs1 = arrayfire::Indexer::default();
        let seq1 = arrayfire::Seq::new(0.0, (space_dims - 1) as f32, 1.0);
        idxrs1.set_index(&randidx, 0, None);
        idxrs1.set_index(&seq1, 1, Some(false));
        dir_line = arrayfire::index_gen(&dir_line, idxrs1);

        line_num = start_line.dims()[0] as u32;

        raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / (start_line.dims()[0])) as u64);

        intersect = arrayfire::constant::<bool>(false, single_dims);

        line_sphere_intersect_batch(
            raytrace_batch_size,
            &start_line,
            &dir_line,
            neuron_pos,
            &circle_radius,
            &mut intersect,
        );

        intersect = arrayfire::reorder_v2(&intersect, 2, 1, Some(vec![0]));

        idx_intersect = arrayfire::locate(&intersect);

        div_idx = arrayfire::div(&idx_intersect, &active_size_u32, false);

        mod_idx = arrayfire::modulo(&idx_intersect, &active_size_u32, false);

        let (key, value) = arrayfire::min_by_key(&div_idx, &mod_idx, 0);

        newWRowIdxCOO = arrayfire::modulo(&key, &cur_num_u32, false).cast::<i32>();

        newWColIdx = value.cast::<i32>();

        newWRowIdxCOO = arrayfire::lookup(&idxsel, &newWRowIdxCOO, 0);

        *WRowIdxCOO = arrayfire::join(0, &newWRowIdxCOO, WRowIdxCOO);

        *WColIdx = arrayfire::join(0, &newWColIdx, WColIdx);

        global_idx = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);

        global_idx = arrayfire::set_unique(&global_idx, false);

        *WRowIdxCOO = arrayfire::div(&global_idx, &neuron_size, false).cast::<i32>();

        *WColIdx = arrayfire::modulo(&global_idx, &neuron_size, false).cast::<i32>();

        if WRowIdxCOO.dims()[0] > con_num {
            break;
        }

        curidxsel = WColIdx.clone();

        curidxsel = find_unique(&curidxsel, neuron_size);
    }
}

/*
Raytracing algorithm 2 for creating neural connections. Connects all neurons within the neural network sphere at the same time

Inputs
raytrace_options:    Raytracing options
netdata:             Network metadata
glia_pos_total:      The positions of all glial cells
input_pos_total:     Selected neurons positions as source for the rays
input_idx_total:     Selected neurons positions as source for the rays
hidden_pos_total:    Selected neurons positions as targets for the rays
hidden_idx_total:    Selected neurons positions as targets for the rays


Outputs:
WRowIdxCOO:     Row vector in the COO sparse matrix
WColIdx:        Column vector in the COO sparse matrix

*/

pub fn RT2_directly_connected(
    raytrace_options: &raytrace_option_type,

    netdata: &network_metadata_type,
    glia_pos_total: &arrayfire::Array<f32>,

    input_pos_total: &arrayfire::Array<f32>,
    input_idx_total: &arrayfire::Array<i32>,

    hidden_pos_total: &arrayfire::Array<f32>,
    hidden_idx_total: &arrayfire::Array<i32>,

    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = input_pos_total.dims()[0];
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f32 = netdata.time_step.clone();
    let nratio: f32 = netdata.nratio.clone();
    let neuron_std: f32 = netdata.neuron_std.clone();
    let sphere_rad: f32 = netdata.sphere_rad.clone();
    let neuron_rad: f32 = netdata.neuron_rad.clone();
    let con_rad: f32 = netdata.con_rad.clone();
    let init_prob: f32 = netdata.init_prob.clone();

    let input_connection_num: u64 = raytrace_options.input_connection_num.clone();
    let max_rounds: u64 = raytrace_options.max_rounds.clone();
    let ray_glia_intersect: bool = raytrace_options.ray_glia_intersect.clone();
    let ray_neuron_intersect: bool = raytrace_options.ray_neuron_intersect.clone();

    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut gidxOld = arrayfire::constant::<u64>(0, single_dims);

    let mut gidxOld_cpu: Vec<u64> = Vec::new();

    let WColIdxelements = WColIdx.elements();

    //let mut WValues_cpu = Vec::new();
    let mut WRowIdxCOO_cpu = Vec::new();
    let mut WColIdx_cpu = Vec::new();

    //Insert previous COO Matrix values
    if WColIdxelements > 1 {
        //TO CPU

        WRowIdxCOO_cpu = vec![i32::default(); WRowIdxCOO.elements()];
        WRowIdxCOO.host(&mut WRowIdxCOO_cpu);

        WColIdx_cpu = vec![i32::default(); WColIdx.elements()];
        WColIdx.host(&mut WColIdx_cpu);

        //Compute global index
        gidxOld = get_global_weight_idx(neuron_size, &WRowIdxCOO, &WColIdx);

        //TO CPU
        gidxOld_cpu = vec![u64::default(); gidxOld.elements()];
        gidxOld.host(&mut gidxOld_cpu);
        drop(gidxOld);

        *WRowIdxCOO = arrayfire::constant::<i32>(0, single_dims);
        *WColIdx = arrayfire::constant::<i32>(0, single_dims);
    }

    let mut hidden_idx = arrayfire::constant::<i32>(0, single_dims);

    let mut tiled_input_idx = arrayfire::constant::<i32>(0, single_dims);

    let mut tiled_hidden_idx = arrayfire::constant::<i32>(0, single_dims);

    //Get input and hidden positions

    let mut hidden_pos = arrayfire::constant::<f32>(0.0, single_dims);

    let mut circle_radius = arrayfire::constant::<f32>(
        neuron_rad,
        arrayfire::Dim4::new(&[hidden_pos.dims()[0], 1, 1, 1]),
    );

    let mut start_line = arrayfire::constant::<f32>(0.0, single_dims);
    let mut dir_line = arrayfire::constant::<f32>(0.0, single_dims);

    let mut raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / start_line.dims()[0]) as u64);

    let mut hidden_size_u32 = hidden_pos_total.dims()[0] as u32;
    let mut hidden_size = hidden_pos_total.dims()[0];

    //Store COO Matrix values

    let mut join_all = Vec::new();

    let mut gidx1 = arrayfire::constant::<u64>(0, single_dims);

    let mut gidx1_cpu: Vec<u64> = Vec::new();

    let mut prev_con_num = 0;

    let mut input_idx_size = 0;

    let mut rng = rand::thread_rng();
    let rand_vec: Vec<u64> = (0..input_size).collect();
    let mut select_input_idx: u64 = 0;

    let mut input_pos = arrayfire::constant::<f32>(0.0, single_dims);
    let mut input_idx = arrayfire::constant::<i32>(0, single_dims);

    let mut glia_pos = arrayfire::constant::<f32>(0.0, single_dims);
    let mut glia_idx = arrayfire::constant::<i32>(0, single_dims);

    let mut same_counter: u64 = 0;

    for vv in 0..max_rounds {
        select_input_idx = rand_vec.choose(&mut rng).unwrap().clone();
        let target_input = arrayfire::row(input_pos_total, select_input_idx as i64);

        input_pos = input_pos_total.clone();
        input_idx = input_idx_total.clone();

        /*
        filter_rays(
            2.0f32*con_rad,

            &target_input,

            &mut input_pos,
            &mut input_idx,
        );
        */

        if input_idx.dims()[0] == 0 {
            continue;
        }

        input_idx_size = input_idx.dims()[0];

        hidden_pos = hidden_pos_total.clone();
        hidden_idx = hidden_idx_total.clone();

        /*
        filter_rays(
            con_rad,

            &target_input,

            &mut hidden_pos,
            &mut hidden_idx,
        );
        */

        hidden_size = hidden_idx.dims()[0];
        hidden_size_u32 = hidden_idx.dims()[0] as u32;

        if hidden_size == 0 {
            continue;
        }

        //Generate rays starting from input neurons
        start_line = arrayfire::constant::<f32>(0.0, single_dims);
        dir_line = arrayfire::constant::<f32>(0.0, single_dims);

        let tile_dims = arrayfire::Dim4::new(&[hidden_size, 1, 1, 1]);

        tiled_input_idx = arrayfire::tile(&input_idx, tile_dims);
        drop(input_idx);

        tiled_hidden_idx = hidden_idx.clone();
        drop(hidden_idx);

        tileDown_i32(input_idx_size, &mut tiled_hidden_idx);

        //println!("z1");
        //println!("input_pos.dims()[0] {}",input_pos.dims()[0]);
        //println!("input_pos.dims()[1] {}",input_pos.dims()[1]);
        //println!("hidden_pos.dims()[0] {}",hidden_pos.dims()[0]);
        //println!("hidden_pos.dims()[1] {}",hidden_pos.dims()[1]);
        //println!("con_rad {}", con_rad);

        rays_from_neuronsA_to_neuronsB(
            con_rad,
            &input_pos,
            &hidden_pos,
            &mut start_line,
            &mut dir_line,
            &mut tiled_input_idx,
            &mut tiled_hidden_idx,
        );
        drop(input_pos);

        //println!("z1");
        //println!("start_line.dims()[0] {}",start_line.dims()[0]);
        //println!("start_line.dims()[1] {}",start_line.dims()[1]);
        //println!("dir_line.dims()[0] {}",dir_line.dims()[0]);
        //println!("dir_line.dims()[1] {}",dir_line.dims()[1]);
        //println!("tiled_input_idx.dims()[0] {}",tiled_input_idx.dims()[0]);
        //println!("tiled_input_idx.dims()[1] {}",tiled_input_idx.dims()[1]);
        //println!("tiled_hidden_idx.dims()[0] {}",tiled_hidden_idx.dims()[0]);
        //println!("tiled_hidden_idx.dims()[1] {}",tiled_hidden_idx.dims()[1]);

        if start_line.dims()[0] == 0 {
            continue;
        }

        raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / start_line.dims()[0]) as u64);

        circle_radius = arrayfire::constant::<f32>(
            neuron_rad,
            arrayfire::Dim4::new(&[hidden_pos.dims()[0], 1, 1, 1]),
        );

        if ray_neuron_intersect && (hidden_size > 1) {
            line_sphere_intersect_batchV2(
                raytrace_batch_size,
                2,
                &hidden_pos,
                &circle_radius,
                &mut start_line,
                &mut dir_line,
                &mut tiled_input_idx,
                &mut tiled_hidden_idx,
            );
        }
        drop(hidden_pos);

        if tiled_input_idx.dims()[0] == 0 {
            continue;
        }

        //println!("a1");
        //println!("intersect.dims()[0] {}",intersect.dims()[0]);
        //println!("intersect.dims()[1] {}",intersect.dims()[1]);
        //println!("intersect.dims()[2] {}",intersect.dims()[2]);

        //println!("a2");
        //println!("intersect.dims()[0] {}",intersect.dims()[0]);
        //println!("intersect.dims()[1] {}",intersect.dims()[1]);
        //println!("intersect.dims()[2] {}",intersect.dims()[2]);

        //println!("input_size {}",input_size);

        glia_pos = glia_pos_total.clone();
        glia_idx = arrayfire::constant::<i32>(0, glia_pos.dims());

        /*
        filter_rays(
            con_rad,

            &target_input,

            &mut glia_pos,
            &mut glia_idx,
        );
        */
        drop(glia_idx);

        raytrace_batch_size = 1 + ((RAYTRACE_LIMIT / start_line.dims()[0]) as u64);

        circle_radius = arrayfire::constant::<f32>(
            neuron_rad,
            arrayfire::Dim4::new(&[glia_pos.dims()[0], 1, 1, 1]),
        );

        if ray_glia_intersect && (glia_pos.dims()[0] > 1) {
            line_sphere_intersect_batchV2(
                raytrace_batch_size,
                0,
                &glia_pos,
                &circle_radius,
                &mut start_line,
                &mut dir_line,
                &mut tiled_input_idx,
                &mut tiled_hidden_idx,
            );
        }
        drop(glia_pos);

        if tiled_input_idx.dims()[0] == 0 {
            continue;
        }

        //Compute global index
        gidx1 = get_global_weight_idx(neuron_size, &tiled_hidden_idx, &tiled_input_idx);

        let mut gidx1_cpu = vec![u64::default(); gidx1.elements()];
        gidx1.host(&mut gidx1_cpu);
        drop(gidx1);

        let mut tiled_hidden_idx_cpu = vec![i32::default(); tiled_hidden_idx.elements()];
        tiled_hidden_idx.host(&mut tiled_hidden_idx_cpu);
        drop(tiled_hidden_idx);

        let mut tiled_input_idx_cpu = vec![i32::default(); tiled_input_idx.elements()];
        tiled_input_idx.host(&mut tiled_input_idx_cpu);
        drop(tiled_input_idx);

        //println!("join_WColIdx.len() {}", join_WColIdx.len());
        //println!("join_WColIdx.keys().len() {}", join_WColIdx.keys().len());
        //println!("gidx1_cpu.len() {}", gidx1_cpu.len());

        //Save new neural connections to COO matrix hashmap
        for qq in 0..gidx1_cpu.len() {
            join_all.push((
                gidx1_cpu[qq].clone(),
                tiled_input_idx_cpu[qq].clone(),
                tiled_hidden_idx_cpu[qq].clone(),
            ));
        }

        //println!("join_WColIdx.len() {}", join_WColIdx.len());
        if ((join_all.len() as u64) > (input_connection_num)) {
            join_all.par_sort_unstable_by_key(|pair| pair.0);
            join_all.dedup_by_key(|pair| pair.0);

            if ((join_all.len() as u64) > (input_connection_num)) {
                break;
            }
        }

        if ((join_all.len() as u64) > prev_con_num) {
            prev_con_num = join_all.len() as u64;
            same_counter = 0;
        } else {
            same_counter = same_counter + 1;
        }

        if same_counter > 5 {
            break;
        }
    }

    drop(gidx1_cpu);
    //Insert previous COO Matrix values
    if WColIdxelements > 1 {
        //Insert old value
        for qq in 0..gidxOld_cpu.len() {
            join_all.push((
                gidxOld_cpu[qq].clone(),
                WColIdx_cpu[qq].clone(),
                WRowIdxCOO_cpu[qq].clone(),
            ));
        }
        drop(gidxOld_cpu);
        //println!("join_WValues.len() {}", join_WValues.len());
    }

    //Sort global index
    join_all.par_sort_unstable_by_key(|pair| pair.0);
    join_all.dedup_by_key(|pair| pair.0);

    let (WColIdx_cpu, WRowIdxCOO_cpu): (Vec<_>, Vec<_>) =
        join_all.par_iter().cloned().map(|(_, b, c)| (b, c)).unzip();

    //Convert cpu vector to gpu array
    *WColIdx = arrayfire::Array::new(
        &WColIdx_cpu,
        arrayfire::Dim4::new(&[WColIdx_cpu.len() as u64, 1, 1, 1]),
    );
    *WRowIdxCOO = arrayfire::Array::new(
        &WRowIdxCOO_cpu,
        arrayfire::Dim4::new(&[WColIdx_cpu.len() as u64, 1, 1, 1]),
    );
}
