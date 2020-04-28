//! This file provides multiclass logistic regression
//! It is used in examples with MNIST data
//! 

extern crate rand;
extern crate rand_distr;

use ndarray::prelude::*;

#[allow(unused_imports)]
use log::Level::*;
#[allow(unused_imports)]
use log::*;

// use log::{debug, info, warn, trace, log_enabled};

use crate::types::*;



//  data dimension have been augmented by one for interception term.
//  The coefficients of the last class have been assumed to be 0 to take into account
//  for the identifiability constraint (Cf Less Than a Single Pass SCSG Lei-Jordan or 
//  Machine Learning Murphy par 9.2.2.1-2
//
pub struct LogisticRegression {
    nbclass : usize,
    // length of observation + 1. Values 1. in slot 0 of arrays.
    observations: Vec<(Array1<f64>, usize)>,
}


impl LogisticRegression {
    pub fn new(nbclass : usize, observations : Vec<(Array1<f64>, usize)>) -> LogisticRegression {
        LogisticRegression {
            nbclass,
            observations,
        }
    } // end of new

} // end of impl LogisticRegression



/// We implement the trait Summation using 2 dimensional Arrays.
/// We use the variable coefficients : Array2<f64> so that 
/// a row is coefficient array corresponding to (1 augmented) observations and a column is coefficients by class.
/// Recall that ndarray is by default with C storage (row oriented)
/// 
impl Summation<Ix2> for LogisticRegression {
    fn terms(&self) -> usize {
        self.observations.len()
    }
    //
    fn term_value(&self, coefficients : &Array2<f64> , term: usize) -> f64 {
        // extract vector i of data and its class
        let (ref x, term_class) = self.observations[term];
        //
        let mut dot_xi = Array1::<f64>::zeros(self.nbclass-1);
        //
        let mut log_arg = 1.0f64;
        for i in 0..self.nbclass-1 {
            // take i row
            assert_eq!(x.len(), coefficients.index_axis(Axis(0),i).len());
            dot_xi[i] = x.dot(&coefficients.slice(s![i, ..]));
            //        we could do
            //        let dot_i = x.dot(&coefficients.index_axis(Axis(0),i));
            log_arg += dot_xi[i].exp();
        }
        // keep term corresponding to term_class (class of term passed as arg)
        let mut other_term = 0.;
        if term_class <  self.nbclass-1 {
            other_term = dot_xi[term_class];
        }
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
        let mut dot_xk = Array1::<f64>::zeros(self.nbclass-1);
        let mut den : f64 = 1.;
        for k in 0..self.nbclass-1 {
            dot_xk[k] = x.dot(&w.slice(s![k, ..])).exp();
            den += dot_xk[k];
        }
        //
        for k in 0..self.nbclass-1 {
            let mut g_term : f64;
            for j in 0..x.len() {
                g_term = x[j] * dot_xk[k]/den;
                // keep term corresponding to term_class (class of term passed as arg)
               if term_class == k {
                    g_term -= x[j];
                }
                gradient[[k, j]] = g_term;
            }
        }
    }  // end of term_gradient
}

