//! logistic regression  and test on Mnit data
//! 
//! 

extern crate env_logger;
extern crate rand;
extern crate rand_distr;
extern crate multistochgrad;

use ndarray::prelude::*;

use rand_distr::{Normal, Distribution};
use rand::random;

use multistochgrad::scsg::*;
use multistochgrad::types::*;



//  data dimension have been augmented by one for interception term.
//  The coefficients of the last class have been assumed to be 0 to take into account
//  for the identifiability constraint (Cf Less Than a Single Pass SCSG Lei-Jordan or 
//  Machine Learning Murphy par 9.2.2.1-2
//
struct ClassificationLogistic {
    nbclass : usize,
    // length of observation + 1. Values 1. in slot 0 of arrays.
    observations: Vec<(Array1<f64>, u64)>,
    // one vector by class for all classes except the last.
    coefficients : Vec<Array1<f64>>
}



impl Summation<Ix1> for ClassificationLogistic {
    fn terms(&self) -> usize {
        self.observations.len()
    }
    //
    fn term_value(&self, coefficients : &Array1<f64> , term: usize) -> f64 {
        let (ref x, term_class) = self.observations[term];
        let mut log_arg = 1.0f64; //////////////!!!
        for i in 0..self.nbclass-1 {
            let dot_i = x.dot(coefficients);
            log_arg += dot_i.exp();
        }
        let other_term = x.dot(coefficients);
        let t_value = log_arg.ln() - other_term;
        t_value
    }
}

impl SummationC1 for SSE {
    fn term_gradient(&self, w: &[f64], i: &usize) -> Vec<f64> {
        let (ref x, y) = self.observations[*i];
        let e = y - linear_regression(w, x);
        let mut gradient = vec![e * -1.0];
        for x in x {
            gradient.push(e * -x);
        }
        gradient
    }
}


/// here x is a data vector , w is 
fn logistic_regression(w: &[f64], x: &[f64]) -> f64 {
    let mut y = w[0];
    for (w, x) in w[1..].iter().zip(x) {
        y += w * x;
    }
    y
}


fn main() {

}  // end of main