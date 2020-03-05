


use log::Level::*;
use log::{debug, info, warn, trace, log_enabled};

use rand::{SeedableRng};

use rand_xoshiro::Xoshiro256PlusPlus;

use crate::types::*;

/// Provides _stochastic_ Gradient Descent optimization
/// as described in Lei-Jordan On the adaptativity of stochastic gradient based optimisation
pub struct StochasticControlledGradientDescent {
    rng: Xoshiro256PlusPlus,
    max_iterations: Option<u64>,
    step_width: f64
}

impl  StochasticControlledGradientDescent {

/// Seeds the random number generator using the supplied `seed`.
/// This is useful to create re-producable results.
    pub fn seed(&mut self, seed: [u8; 32]) {
        self.rng = Xoshiro256PlusPlus::from_seed(seed);
    }
    
} // end impl StochasticControlledGradientDescent



impl<F: Summation1> Minimizer<F> for  StochasticControlledGradientDescent {
    type Solution = Solution;

    fn minimize(&self, function: &F, initial_position: Vec<f64>) -> Solution {
        let mut position = initial_position;
        let mut value = function.value(&position);

        if log_enabled!(Trace) {
            info!("Starting with y = {:?} for x = {:?}", value, position);
        } else {
            info!("Starting with y = {:?}", value);
        }

        let mut iteration = 0;
        let mut terms: Vec<_> = (0..function.terms()).collect();
        let mut rng = self.rng.clone();

        loop {

            for batch in terms.chunks(self.mini_batch) {
                let gradient = function.partial_gradient(&position, batch);

                // step into the direction of the negative gradient
                for (x, g) in position.iter_mut().zip(gradient) {
                    *x -= self.step_width * g;
                }
            }

            value = function.value(&position);

            iteration += 1;

            if log_enabled!(Trace) {
                debug!("Iteration {:6}: y = {:?}, x = {:?}", iteration, value, position);
            } else {
                debug!("Iteration {:6}: y = {:?}", iteration, value);
            }

            let reached_max_iterations = self.max_iterations.map_or(false,
                |max_iterations| iteration == max_iterations);

            if reached_max_iterations {
                info!("Reached maximal number of iterations, stopping optimization");
                return Solution::new(position, value);
            }
        }
    } // end minimize
}  // end impl impl<F: Summation1> Minimizer<F>
