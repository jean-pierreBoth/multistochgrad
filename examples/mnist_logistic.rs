//! This file provides an example of logisic regression on MNIST digits database



extern crate env_logger;
extern crate rand;
extern crate rand_distr;
extern crate multistochgrad;

use std::path::{PathBuf};
use std::fs::{OpenOptions};


use ndarray::prelude::*;


use multistochgrad::prelude::*;
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
                image[index] = images[[i,j,k]] as f64;
            }
        } // end of for i
        observations.push((image, labels[k] as usize));
    }  // end of for k
    //
    let regr_l = LogisticRegression::new(10, observations);
    //
    // minimize
    //
    // get image of coefficients to see

}