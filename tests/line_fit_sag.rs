// Copyright (c) 2016 Oliver Mader <b52@reaktor42.de>
//
// This file is taken from the crate optimisation, with minor modifications,

//! Illustration of fitting a linear regression model using stochastic gradient descent
//! given a few noisy sample observations.
//!
//! Run with `RUST_LOG=trace    cargo test --test line_fitting`.


extern crate env_logger;
extern crate rand;
extern crate rand_distr;
extern crate multistochgrad;


use rand_distr::{Normal, Distribution};
use rand::random;

use ndarray::prelude::*;


use multistochgrad::sag::*;
use multistochgrad::types::*;

use multistochgrad::applis::linear_regression::*;

#[test]
fn test_line_regression() {
    let _ = env_logger::init();

    log::set_max_level(log::LevelFilter::Trace);
    // the true coefficients of our linear model
    let true_coefficients = vec![13.37, -4.2, 3.14];

    println!("Trying to approximate the true linear regression coefficients {:?} using SGD \
        given 100 noisy samples", true_coefficients);

    let true_coefficients_arr = Array1::<f64>::from(true_coefficients);
    let normal = Normal::new(0., 1.).unwrap();
    let noisy_observations = (0..100).map(|_| {
        let mut v = Vec::<f64>::with_capacity(3);
        v.push(1.);
        v.push(random::<f64>());
        v.push(random::<f64>());
        let x = Array1::<f64>::from(v);
        let y = linear_regression(&true_coefficients_arr, &x) +
                normal.sample(&mut rand::thread_rng());
        (x.iter().cloned().collect(), y)
    }).collect();


    // the actual function we want to minimize, which in our case corresponds to the
    // sum squared error
    let sse = SSE {
        observations: noisy_observations
    };
    // eta_0, m_0  factor , b_0  , B_0 factor 
    let sag_pb = SagDescent::new(0.5 ,   // step size
                                    );
    //
    let initial_position = Array1::<f64>::from( vec![1.0; true_coefficients_arr.len()]);
    let nb_iter = 10000;
    let solution = sag_pb.minimize(&sse, &initial_position, nb_iter);

    println!(" solution with a SSE = {:2.4E}", solution.value);
    for i in 0..solution.position.len() {
        println!("{:2.4E} ", solution.position[i]);
    }
    assert!(solution.value < 0.1);
}
