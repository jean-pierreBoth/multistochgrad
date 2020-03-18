


use log::Level::*;
use log::{debug, info, warn, trace, log_enabled};

use rand::{SeedableRng};
use rand::distributions::{Distribution};
// a fast but non crypto secure algo. method jump to use in // !!!!
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::types::*;

/// Provides _stochastic_ Gradient Descent optimization
/// as described in Lei-Jordan On the adaptativity of stochastic gradient based optimisation 2019
pub struct StochasticControlledGradientDescent {
    rng: Xoshiro256PlusPlus,
    mini_batch_size: usize,
    // mj parameter of paper
    inner_loop_size : usize,
    // Bj in Paper
    large_batch_size: usize,
    // T in paper.
    max_iterations: Option<usize>,
    momentum : Option<f64>,
    step_width: f64
}

impl  StochasticControlledGradientDescent {

/// Seeds the random number generator using the supplied `seed`.
/// This is useful to create re-producable results.
    pub fn seed(&mut self, seed: [u8; 32]) {
        self.rng = Xoshiro256PlusPlus::from_seed(seed);
    }
    /// sample number of mini batch according to geometric law of parameter p = b_j/(m_j+b_j) 
    /// with law : P(N=k) = (1-p) * p^k. (proba of nb trial before success mode)
    fn sample_nb_small_mini_batches(&self, j : usize, rng : &mut Xoshiro256PlusPlus) -> usize {
        let mut m_batch_size_j = 1;
        let m_j = self.inner_loop_size as f64;
        let b_j = self.mini_batch_size as f64;
        let p : f64 = m_j/(m_j+b_j);
        // pass a &mut here!!
        let mut xsi : f64 = rand_distr::Standard.sample(rng);
        // success is when xsi is >= p !!
        while xsi < p {
            xsi = rand_distr::Standard.sample(rng);
            m_batch_size_j += 1;
        }
        trace!(" mini batch size {:?} ", m_batch_size_j);
        return m_batch_size_j;
    }
    // sample (without replacemet) size terms among in_terms
    // should have size of in_terms >>  size
    fn sample_batch_terms(&self, size: usize, in_terms: &[usize], rng : &mut Xoshiro256PlusPlus) -> Vec<usize> {
        let mut out_terms = Vec::<usize>::with_capacity(size);
        //
        assert!(size >= in_terms.len());

        // sample terms. Cf Knuth The Art of Computer Programming, Volume 2, Section 3.4.2 
        // https://bastian.rieck.me/blog/posts/2017/selection_sampling/
        let mut t : usize = 0;
        let mut xsi : f64;
        while t < size {
            xsi = rand_distr::Standard.sample(rng);
            if xsi * ((in_terms.len() - t) as f64) < (size-out_terms.len()) as f64 {
                out_terms.push(in_terms[t]);
            }
            t+=1;
        }
        //
        assert_eq!(size, out_terms.len());
        //
        out_terms
    }
    
} // end impl StochasticControlledGradientDescent



impl<F: SummationC1> Minimizer<F> for  StochasticControlledGradientDescent {
    type Solution = Solution;

    fn minimize(&self, function: &F, initial_position: Vec<f64>) -> Solution {
        let mut position = initial_position;
        let mut value = function.value(&position);

        if log_enabled!(Info) {
            info!("Starting with y = {:?} for x = {:?}", value, position);
        } else {
            info!("Starting with y = {:?}", value);
        }

        let mut iteration : usize = 0;
        let mut all_terms: Vec<_> = (0..function.terms()).collect();
        let mut rng = self.rng.clone();

        loop {

            // sample large batch of size Bj


            // sample binomial law for number Nj of small batch iterations
            let n_j = self.sample_nb_small_mini_batches(iteration, &mut rng);
            // loop on small batch iterations

            for _k in 0..n_j {
                // sample mini batch
                let terms = self.sample_batch_terms(n_j, &all_terms, &mut rng);
                let gradient = function.partial_gradient(&position, terms);

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
