// Copyright (c) 2016 Oliver Mader <b52@reaktor42.de>
//
// This file is taken from the crate optimisation, with minor modifications,

//! Illustration of fitting a linear regression model using stochastic gradient descent
//! given a few noisy sample observations.
//!
//! Run with `cargo test --test line_fitting`.


extern crate env_logger;
extern crate rand;
extern crate rand_distr;
extern crate multistochgrad;


use rand_distr::{Normal, Distribution};
use rand::random;

use multistochgrad::scsg::*;
use multistochgrad::types::*;

#[test]
fn test_line_regression() {
    let _ = env_logger::init();

    // the true coefficients of our linear model
    let true_coefficients = &[13.37, -4.2, 3.14];

    println!("Trying to approximate the true linear regression coefficients {:?} using SGD \
        given 100 noisy samples", true_coefficients);

    let normal = Normal::new(0., 1.).unwrap();
    let noisy_observations = (0..100).map(|_| {
        let x = random::<[f64; 2]>();
        let y = linear_regression(true_coefficients, &x) +
                normal.sample(&mut rand::thread_rng());
        (x.iter().cloned().collect(), y)
    }).collect();


    // the actual function we want to minimize, which in our case corresponds to the
    // sum squared error
    let sse = SSE {
        observations: noisy_observations
    };

    let solution = StochasticControlledGradientDescent::new(1., 10, 10, 1.1)
        .minimize(&sse, vec![1.0; true_coefficients.len()]);

    println!("Found coefficients {:?} with a SSE = {:?}", solution.position, solution.value);
    assert_eq!(1,0);
}


// the sum squared error measure we want to minimize over a set of observations
struct SSE {
    observations: Vec<(Vec<f64>, f64)>
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
