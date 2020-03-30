//! This file provides an example of logisic regression on MNIST digits database



extern crate env_logger;
extern crate rand;
extern crate rand_distr;
extern crate multistochgrad;


use rand_distr::{Normal, Distribution};
use rand::random;

use ndarray::prelude::*;

use ndarray::{Array, Zip};

use multistochgrad::prelude::*;

const image_fname = String::from("/home.1/jpboth/Data/MNIST/train-images-idx3-ubyte");
const label_fname = String::from("/home.1/jpboth/Data/MNIST/train-labels-idx1-ubyte");


fn main () {

    // check for path image and labels
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(&image_path);
    if image_file_res.is_err() {
        println!("could not open image file : {:?}", image_fname);
        return;
    }    
    let label_path = PathBuf::from(label_fname.clone());
    let label_file_res = OpenOptions::new().read(true).open(&label_path);
    if label_file_res.is_err() {
        println!("could not open label file : {:?}", label_fname);
        return;
    }
    // load mnist data
    let mnist_data = MnistData(image_fname, label_fname).unwrap();
    // transform into logisitc regression

    // minimize

    // get image of coefficients to see

}