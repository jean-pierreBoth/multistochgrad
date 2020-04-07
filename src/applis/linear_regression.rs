//! linear regression case
//! 
//

use ndarray::prelude::*;
use ndarray::{Array, Zip};

use crate::types::*;

/// the sum squared error measure we want to minimize over a set of observations
pub struct SSE<Ix1> {
    pub observations: Vec<(Array<f64, Ix1>, f64)>
}

impl Summation<Ix1> for SSE<Ix1> {
    fn terms(&self) -> usize {
        self.observations.len()
    }
    fn term_value(&self, w: &Array1<f64>, i: usize) -> f64 {
        let (ref x, y) = self.observations[i];
        0.5 * (y - linear_regression(w, x)).powi(2)
    }
}

impl SummationC1<Ix1> for SSE<Ix1> {
    fn term_gradient(&self, w: &Array1<f64>, i: &usize, gradient : &mut Array1<f64>)  {
        let (ref x, y) = self.observations[*i];
        let e = y - linear_regression(w, x);
        // gradient is -e * x. par_apply uses rayon.
        Zip::from(gradient).and(x).par_apply(| g, &xv| *g = xv * (-e));
    }
}


// a simple linear regression model, i.e., f(x) = w_0 * x_0 + w_1*x_1 + w_2*x_2 + ...
// with x_0 set to 1. !!
/// linear prevision with w as  coeff and x observations 
pub fn linear_regression(w: &Array1<f64> , x: &Array1<f64>) -> f64 {
    assert_eq!(w.len() , x.len());
    let y = w.dot(x);
    y
}
