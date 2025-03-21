extern crate arrayfire;

use std::collections::HashMap;

/*
extern crate mpi;

use mpi::datatype::{MutView, UserDatatype, View};
use mpi::traits::*;
use mpi::Count;

use mpi::traits::Equivalence;





const NAME_SIZE: usize = 256;


#[derive(Equivalence)]
pub struct host_info_type {
    pub hostname: [u8; NAME_SIZE],
    pub hostnamelen: u64,
    pub CUDA_devices: u64,
    pub OPENCL_devices: u64,
    pub CPU_devices: u64,
    pub thread_id: u64
}






pub enum distribute_type {
    MIRROR,
    EVOLUTION
}





pub struct host_config_type {
    pub hostname: String,
    pub backend_vec: Vec<arrayfire::Backend>,
    pub device_id_vec: Vec<i32>,
    pub worker_num_vec: Vec<u64>,
    pub distribute_vec: Vec<distribute_type>
}



pub fn string_to_u8_arr(
    input: String
) -> [u8; NAME_SIZE]
{
    let mut output: [u8; NAME_SIZE] = [0; NAME_SIZE];

    for (zz, c) in input.bytes().enumerate()
    {
        if zz >= (NAME_SIZE)
        {
            break;
        }
        output[zz] = c.clone();
    }

    output
}






pub fn create_host_info_buffer(
    MPI_size: u64
) -> Vec<host_info_type>
{
    let mut host_info_vec: Vec<host_info_type> = Vec::new();

    for qq in 0..MPI_size
    {
        let hostname = String::from("dummy");
        let hostnamebyte = string_to_u8_arr(hostname.clone());
        let host_test = host_info_type {
            hostname: hostnamebyte,
            hostnamelen: hostname.len() as u64,
            CUDA_devices: 0,
            OPENCL_devices: 0,
            CPU_devices: 0,
            thread_id: 0
        };
        host_info_vec.push(host_test);
    }

    host_info_vec
}











pub fn get_host_info(
    thread_id: u64
) -> host_info_type
{

    let hostname = (hostname::get().unwrap()).to_string_lossy().to_string();

    let backend_vec = arrayfire::get_available_backends();


    let mut CUDA_devices = 0;
    let mut OPENCL_devices = 0;
    let mut CPU_devices = 0;



    for back in backend_vec.iter()
    {
        arrayfire::set_backend(*back);
        let device_num = arrayfire::device_count() as u64;

        if back.clone() == arrayfire::Backend::CUDA
        {
            CUDA_devices = device_num;
        }

        if back.clone() == arrayfire::Backend::OPENCL
        {
            OPENCL_devices = device_num;
        }

        if back.clone() == arrayfire::Backend::CPU
        {
            CPU_devices = device_num;
        }


        /*
        for i in 0..device_num
        {
            arrayfire::set_device(i);
            arrayfire::info();
        }
        */
    }

    let hostnamebyte = string_to_u8_arr(hostname.clone());

    let mut hostnamelen = hostname.len() as u64;

    if (hostnamelen > (NAME_SIZE as u64))
    {
        hostnamelen = (NAME_SIZE as u64);
    }


    let host_info = host_info_type {
        hostname: hostnamebyte.clone(),
        hostnamelen: hostnamelen,
        CUDA_devices: CUDA_devices,
        OPENCL_devices: OPENCL_devices,
        CPU_devices: CPU_devices,
        thread_id: thread_id.clone()
    };



    host_info
}








pub fn parse_host_info(
    host_config_hash: &HashMap<String, host_config_type>,
    host_info_vec: &Vec<host_info_type>,


    host_to_thread_map: &mut HashMap<String, Vec<u64> > ,
    thread_to_device_map: &mut HashMap<String, Vec<Vec<u64> >  >
)
{

    for hostpointer in host_config_hash.keys()
    {
        let hostname = hostpointer.clone();

        //let hostname = config.hostname.clone();

        let mut cur_thread_vec: Vec<u64> = Vec::new();

        for info in host_info_vec.iter()
        {

            let mut cur_hostname: String = String::from_utf8(info.hostname.clone().to_vec()).unwrap();

            cur_hostname = cur_hostname[..(info.hostnamelen as usize)].to_string();


            if (cur_hostname != hostname)
            {
                continue;
            }

            cur_thread_vec.push(info.thread_id.clone() );
        }

        cur_thread_vec.sort();

        host_to_thread_map.insert(hostname.clone(), cur_thread_vec.clone() );







        let worker_num_vec =  host_config_hash[&hostname].worker_num_vec.clone();

        let mut start_idx = 0;
        let mut end_idx = 0;

        let mut thread_group_vec: Vec<Vec<u64> > = Vec::new();

        for worker_num in worker_num_vec.iter()
        {
            end_idx = start_idx + (worker_num.clone() as usize);
            let collect_threads = cur_thread_vec[start_idx..end_idx].to_vec();

            thread_group_vec.push(collect_threads);

            start_idx = end_idx;
        }


        thread_to_device_map.insert(hostname.clone(), thread_group_vec.clone() );

    }






}






pub fn set_host_config(
    thread_id: u64,
    host_config_hash: &HashMap<String, host_config_type>,
    thread_to_device_map: &HashMap<String, Vec<Vec<u64> >  >,

    backend_out: &mut arrayfire::Backend,
    device_id_out: &mut i32
) -> bool
{
    let hostname = (hostname::get().unwrap()).to_string_lossy().to_string();

    let thread_vec = thread_to_device_map[&hostname].clone();


    let mut found = false;
    let mut idx = 0;
    for ii in 0..thread_vec.len()
    {
        if (thread_vec[ii].contains(&thread_id))
        {
            found = true;
            idx = ii;
            break;
        }
    }

    if (found)
    {
        let backend_vec = host_config_hash[&hostname].backend_vec.clone();
        let device_id_vec = host_config_hash[&hostname].device_id_vec.clone();

        arrayfire::set_backend(backend_vec[idx].clone());
        arrayfire::set_device(device_id_vec[idx].clone());
        arrayfire::sync(device_id_vec[idx].clone());

        *backend_out = backend_vec[idx].clone();
        *device_id_out = device_id_vec[idx].clone();
    }


    found
}


*/
