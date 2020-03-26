//! logistic regression  and test on Mnit data
//! 
//! 

extern crate env_logger;
extern crate rand;
extern crate rand_distr;
extern crate multistochgrad;

use ndarray::{Array3, Array1, s};

use rand_distr::{Normal, Distribution};
use rand::random;

use multistochgrad::scsg::*;
use multistochgrad::types::*;



// the sum squared error measure we want to minimize over a set of observations
// value of observation is discrete
struct ClassifData {
    observations: Vec<(Array1<f64>, u64)>
}

impl Summation for SSE {
    fn terms(&self) -> usize {
        self.observations.len()
    }
    fn term_value(&self, w: &[f64], i: usize) -> f64 {
        let (ref x, y) = self.observations[i];
        0.5 * (y - linear_regression(w, x)).powi(2)
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


// a simple linear regression model, i.e., f(x) = w_0 + w_1*x_1 + w_2*x_2 + ...
fn linear_regression(w: &[f64], x: &[f64]) -> f64 {
    let mut y = w[0];
    for (w, x) in w[1..].iter().zip(x) {
        y += w * x;
    }
    y
}
