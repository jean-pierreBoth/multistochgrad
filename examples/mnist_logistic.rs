//! This file provides an example of logisic regression on MNIST digits database



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
use multistochgrad::logistic_regression::*;


const IMAGE_FNAME_STR : &str = "/home.1/jpboth/Data/MNIST/train-images-idx3-ubyte";
const LABEL_FNAME_STR : &str = "/home.1/jpboth/Data/MNIST/train-labels-idx1-ubyte";


fn main () {

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
    // transform into logisitc regression
    let mut observations = Vec::<(Array1<f64>, usize)>::with_capacity(10);
    // nb_images is length of third compoenent of array dimension
    let (nb_row, nb_column, nb_images) = images.dim();   // get t-uple from dim method
    assert_eq!(nb_images, labels.shape()[0]);            // get slice from shape method...
    //
    for k in 0..nb_images {
        let mut image = Array1::<f64>::zeros(1+nb_row*nb_column);
        let index = 0;
        for i in 0..nb_row {
            for j in 0..nb_column {
                image[index] = images[[i,j,k]] as f64/256.;
            }
        } // end of for i
        observations.push((image, labels[k] as usize));
    }  // end of for k
    //
    let regr_l = LogisticRegression::new(10, observations);
    //
    // minimize
    //
       // m_0, b_0 , B_0, alfa
    let nb_iter = 100;
    let scgd_pb = StochasticControlledGradientDescent::new(1., 1, 100, 1.1);
    // allocate and set to 0 an array with 9 rows(each row corresponds to a class, columns are pixels values)
    let initial_position = Array2::<f64>::zeros((9, 1+nb_row*nb_column));
    
    let solution = scgd_pb.minimize(&regr_l, &initial_position , nb_iter);
    println!(" solution with a SSE = {:2.4E}", solution.value);
    //
    // get image of coefficients to see corresponding images.
    //
    let image_fname = String::from("classe.img");
    for k in 0..9 {

        let mut k_image_fname : String = image_fname.clone();
        k_image_fname.push(k as u8 as char);
        let image_path = PathBuf::from(k_image_fname.clone());
        let image_file_res = OpenOptions::new().write(true).open(&image_path);
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