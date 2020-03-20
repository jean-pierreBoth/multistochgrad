


use log::Level::*;
use log::{debug, info, warn, trace, log_enabled};

use rand::{SeedableRng};
use rand::distributions::{Distribution};
// a fast but non crypto secure algo. method jump to use in // !!!!
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::types::*;

// a struct to have batch size (large and mini) to use at a given iteration
// we must have large_batch >= nb_mini_batch_parameter >= mini_batch  (Cf Th 4.1)
pub struct  BatchSizeInfo {
    step : usize,
    // large batch size
    large_batch : usize,
    // mini batvh size
    mini_batch: usize,
    // nb mini bath parameter
    nb_mini_batch_parameter : f64,   
}



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
    // governs growing sizes of B_j and m_j (large and mini batch sizes)
    batch_growing_factor : f64,
    step_width: f64
}

impl  StochasticControlledGradientDescent {

    /// Seeds the random number generator using the supplied `seed`.
    /// This is useful to create re-producable results.
    pub fn seed(&mut self, seed: [u8; 32]) {
        self.rng = Xoshiro256PlusPlus::from_seed(seed);
    }
    ///
    pub fn get_batch_size_at_jstep(&self, j: usize) -> BatchSizeInfo {
        let alfa_j = self.batch_growing_factor.powi(j as i32);
        let mini_batch_size = 10;
        BatchSizeInfo {
            step : j,
            large_batch : (alfa_j * alfa_j).ceil() as usize,
            mini_batch : mini_batch_size,
            nb_mini_batch_parameter : alfa_j,
        }
    } // end of get_batch_size_at_jstep
    ///
    fn get_step_size_at_jstep(&self, j:usize) -> f64 {
        let step_size = 1./(1+j) as f64;
        step_size
    }
    /// 
    /// sample number of mini batch according to geometric law of parameter p = b_j/(m_j+b_j) 
    /// with law : P(N=k) = (1-p) * p^k. (proba of nb trial before success mode)
    fn sample_nb_small_mini_batches(&self, j : usize, rng : &mut Xoshiro256PlusPlus) -> usize {
        let mut m_batch_size_j = 1;
        let batch_size_info : BatchSizeInfo = self.get_batch_size_at_jstep(j);
        let m_j =  batch_size_info.nb_mini_batch_parameter as f64;
        let b_j = batch_size_info.mini_batch as f64;
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
    
} // end impl StochasticControlledGradientDescent



fn sample_without_replacement_from_slice(size_asked: usize, in_terms: &[usize], rng : &mut Xoshiro256PlusPlus) -> Vec<usize> {
        let mut out_terms = Vec::<usize>::with_capacity(size_asked);
        // sample terms. Cf Knuth The Art of Computer Programming, Volume 2, Section 3.4.2 
        // https://bastian.rieck.me/blog/posts/2017/selection_sampling/
        let mut t : usize = 0;
        let mut xsi : f64;
        
        while t < size_asked {
            xsi = rand_distr::Standard.sample(rng);
            if xsi * ((in_terms.len() - t) as f64) < (size_asked - out_terms.len()) as f64 {
                out_terms.push(in_terms[t]);
            }
            t+=1;
        }
        //
        assert_eq!(size_asked, out_terms.len());
        //
        out_terms
    }

// this functio requires that size_in be equal to in_temrs.count() !!!!
// but it can be useful as it enables call with a range and thus avoid passing reference to large slice!
fn sample_without_replacement_iter(size_asked: usize, in_terms: impl IntoIterator<Item=usize>, size_in : usize, rng : &mut Xoshiro256PlusPlus) -> Vec<usize> {

    let mut out_terms = Vec::<usize>::with_capacity(size_asked);
    let mut xsi : f64;
    for t in  in_terms.into_iter() {
        xsi = rand_distr::Standard.sample(rng);
        if (xsi * ((size_in - t) as f64)) < (size_asked-out_terms.len()) as f64 {
            out_terms.push(t);
        }
    }  
    out_terms 
}  // end of sample_without_replacement



impl<F: SummationC1> Minimizer<F> for  StochasticControlledGradientDescent {
    type Solution = Solution;

    fn minimize(&self, function: &F, initial_position: Vec<f64>) -> Solution {
        let mut position = initial_position;
        let dimension = position.len();
        let mut value = function.value(&position);
        let mut direction : Vec<f64> = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            direction.push(0.);
        } 

        if log_enabled!(Info) {
            info!("Starting with y = {:?} for x = {:?}", value, position);
        } else {
            info!("Starting with y = {:?}", value);
        }

        let mut iteration : usize = 0;
        let all_terms: Vec<_> = (0..function.terms()).collect();
        let mut rng = self.rng.clone();
        let nb_terms = function.terms();

        loop {
            // get iteration parameters
            let iter_params = self.get_batch_size_at_jstep(iteration);

            // sample large batch of size Bj
            let large_batch_indexes = sample_without_replacement_iter(iter_params.large_batch, 0..nb_terms, nb_terms, & mut rng);
            // compute gradient on large batch index set and store initial position
            let large_batch_gradient = function.partial_gradient(&position, &large_batch_indexes);
            let position_before_mini_batch = position.clone();
            let mut position_during_mini_batches = position.clone();
            // sample binomial law for number Nj of small batch iterations
            let n_j = self.sample_nb_small_mini_batches(iteration, &mut rng);
            // loop on small batch iterations
            for _k in 0..n_j {
                // sample mini batch terms
                let terms = sample_without_replacement_iter(n_j, 0..nb_terms, nb_terms, &mut rng);
                let mini_batch_gradient_current = function.partial_gradient(&position_during_mini_batches, &terms);
                let mini_batch_gradient_origin = function.partial_gradient(&position_before_mini_batch, &terms);
                for i in 0..dimension {
                    direction[i] = mini_batch_gradient_current[i] - mini_batch_gradient_origin[i] + large_batch_gradient[i];
                }
                // step into the direction of the negative gradient
                for i in 0..dimension  {
                    position_during_mini_batches[i] -= self.get_step_size_at_jstep(iteration) * direction[i];
                }
            } // end mini batch loop
            // update position
            for i in 0..dimension {
                position[i] = position_during_mini_batches[i];
            }
            iteration += 1;

            value = function.value(&position);
            if log_enabled!(Trace) {
                debug!("Iteration {:6}: y = {:?}, x = {:?}", iteration, value, position);
            } else {
                debug!("Iteration {:6}: y = {:?}", iteration, value);
            }
            // convergence control or max iterations control
            if let Some(max_iterations) = self.max_iterations {
                if iteration >= max_iterations {
                    info!("Reached maximal number of iterations required , stopping optimization");
                    return Solution::new(position, value);
                }
            }
            else {
                info!("Reached default number  of iterations required  {:?} , stopping optimization", 100);
                return Solution::new(position, value);
            }
        } // end global loop

    } // end minimize
}  // end impl impl<F: Summation1> Minimizer<F>
