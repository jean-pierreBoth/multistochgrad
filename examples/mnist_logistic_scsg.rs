//! This file provides an example of logisic regression on MNIST digits database
//! Download mnit data base from http://yann.lecun.com/exdb/mnist/
//! Change file name data base to your settings.
///
/// to run with cargo run --release --example mnist_regression
/// or with RUST_LOG=debug|info cargo run --example mnist_logistic_scsg
/// 

extern crate env_logger;
extern crate rand;
extern crate rand_distr;
extern crate multistochgrad;

use std::path::{PathBuf};
use std::fs::{OpenOptions};

use std::io;
use std::io::prelude::*;

use ndarray::prelude::*;

use multistochgrad::prelude::*;

//use multistochgrad::prelude::*;
use multistochgrad::applis::logistic_regression::*;


const IMAGE_FNAME_STR : &str = "/home.1/jpboth/Data/MNIST/train-images-idx3-ubyte";
const LABEL_FNAME_STR : &str = "/home.1/jpboth/Data/MNIST/train-labels-idx1-ubyte";


fn main () {
    let _ = env_logger::init();
    log::set_max_level(log::LevelFilter::Trace);

    // check for path image and labels
    let image_path = PathBuf::from(String::from(IMAGE_FNAME_STR).clone());
    let image_file_res = OpenOptions::new().read(true).open(&image_path);
    if image_file_res.is_err() {
        println!("could not open image file : {:?}", IMAGE_FNAME_STR);
        return;
    }    
    let label_path = PathBuf::from(LABEL_FNAME_STR.clone());
    let label_file_res = OpenOptions::new().read(true).open(&label_path);
    if label_file_res.is_err() {
        println!("could not open label file : {:?}", LABEL_FNAME_STR);
        return;
    }
    //
    // load mnist data
    //
    let mnist_data = MnistData::new(String::from(IMAGE_FNAME_STR), String::from(LABEL_FNAME_STR)).unwrap();
    let images = mnist_data.get_images();
    let labels = mnist_data.get_labels();
    // nb_images is length of third compoenent of array dimension
    let (nb_row, nb_column, nb_images) = images.dim();   // get t-uple from dim method
    assert_eq!(nb_images, labels.shape()[0]);            // get slice from shape method...
    // transform into logisitc regression
    let mut observations = Vec::<(Array1<f64>, usize)>::with_capacity(nb_images);
    //
    for k in 0..nb_images {
        let mut image = Array1::<f64>::zeros(1+nb_row*nb_column);
        let mut index = 0;
        image[index] = 1.;
        index += 1;
        for i in 0..nb_row {
            for j in 0..nb_column {
                image[index] = images[[i,j,k]] as f64/256.;
                index += 1;
            }
        } // end of for i
        observations.push((image, labels[k] as usize));
    }  // end of for k
    //
    let regr_l = LogisticRegression::new(10, observations);
    //
    // minimize
    //
    // step, m_0, b_0 , B_0
    let scgd_pb = StochasticControlledGradientDescent::new(0.1, 
                0.004,        // base factor for number of mini batch
                 1,           // base for size of mini batch
                0.02);       // base for large batch size
    // allocate and set to 0 an array with 9 rows(each row corresponds to a class, columns are pixels values)
    let mut initial_position = Array2::<f64>::zeros((9, 1+nb_row*nb_column));
    // do a bad initializion , fill with 0 is much better!!
    initial_position.fill(0.5);
    //
    let nb_iter = 100;
    let solution = scgd_pb.minimize(&regr_l, &initial_position , Some(nb_iter));
    println!(" solution with minimized value = {:2.4E}", solution.value);
    //
    // get image of coefficients to see corresponding images.
    //
    let image_fname = String::from("classe_scsg.img");
    for k in 0..9 {

        let mut k_image_fname : String = image_fname.clone();
        k_image_fname.push_str(&k.to_string());
        let image_path = PathBuf::from(k_image_fname.clone());
        let image_file_res = OpenOptions::new().write(true).create(true).open(&image_path);
        if image_file_res.is_err() {
            println!("could not open image file : {:?}", k_image_fname);
            return;
        }
        // 
        let mut out = io::BufWriter::new(image_file_res.unwrap());
        // 
        // get a f64 slice to write
        let f64_array_to_write : &[f64]  = solution.position.slice(s![k, ..]).to_slice().unwrap();
        let u8_slice = unsafe {  std::slice::from_raw_parts(f64_array_to_write.as_ptr() as *const u8, 
                                            std::mem::size_of::<f64>() * f64_array_to_write.len())
           };
        out.write_all(u8_slice).unwrap();
        out.flush().unwrap();
     //   out.write(&solution.position.slice(s![k, ..])).unwrap();
    }
}