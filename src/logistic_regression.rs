//! This file provides multiclass logistic regression
//! It is used in examples with MNIST data
//! 

extern crate rand;
extern crate rand_distr;

use ndarray::prelude::*;

use crate::types::*;



//  data dimension have been augmented by one for interception term.
//  The coefficients of the last class have been assumed to be 0 to take into account
//  for the identifiability constraint (Cf Less Than a Single Pass SCSG Lei-Jordan or 
//  Machine Learning Murphy par 9.2.2.1-2
//
struct LogisticRegression {
    nbclass : usize,
    // length of observation + 1. Values 1. in slot 0 of arrays.
    observations: Vec<(Array1<f64>, u64)>,
    /// As ndarray is by default with C storage (row oriented) we use the following scheme for storing data
    /// a row is coefficient array corresponding to (1 augmented observations)
    /// a column is coefficients by class.
    coefficients : Array2<f64>
}



impl Summation<Ix2> for LogisticRegression {
    fn terms(&self) -> usize {
        self.observations.len()
    }
    //
    fn term_value(&self, coefficients : &Array2<f64> , term: usize) -> f64 {
        // extract vector i of data and its class
        let (ref x, term_class) = self.observations[term];
        let mut log_arg = 1.0f64;
        for i in 0..self.nbclass-1 {
            // take i row
            assert_eq!(x.len(), coefficients.index_axis(Axis(0),i).len());
            let dot_i = x.dot(&coefficients.slice(s![i, ..]));
            //        we could do
            //        let dot_i = x.dot(&coefficients.index_axis(Axis(0),i));
            log_arg += dot_i.exp();
        }
        let other_term = x.dot(&coefficients.slice(s![term, ..]));
        let t_value = log_arg.ln() - other_term;
        t_value
    } // end of term_value
}

impl SummationC1<Ix2> for LogisticRegression {
    // gradient has in a row  coefficients of class k, q row has a numboer of columns equal to 
    // length of observations.
    fn term_gradient(&self, w: &Array2<f64>, term: &usize, gradient : &mut Array2<f64>) {
        // get observation corresponding to term
        let (ref x, term_class) = self.observations[*term];
        //
        let mut den : f64 = 1.;
        for k in 0..self.nbclass-1 {
            let dot_k = x.dot(&w.slice(s![k, ..]));
            den += dot_k.exp();
        }
        //

        for k in 0..self.nbclass-1 {
            let mut g_term : f64;
            let dot_k : f64 = x.dot(&w.slice(s![k, ..]));
            for j in 0..x.len() {
                g_term = x[j] * dot_k/den;
                if *term == k {
                    g_term -= x[j] * w[[k,j]];
                } 
                gradient[[k, j]] = g_term;
            }
        }
    }  // end of term_gradient
}


