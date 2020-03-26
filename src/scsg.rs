//! A Rust implementation of Lei-Jordan paper 
//! On the adaptativity of Stochastic gradient based optiimization 

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
    _step : usize,
    // large batch size
    large_batch : usize,
    // mini batvh size
    mini_batch: usize,
    // nb mini bath parameter
    nb_mini_batch_parameter : f64,   
}


// m_zero should be around 2 * mini_batch_size_init 
// we try mini batch size m_zero * alfa^j 
//        large batch size large_batch_size_init * alfa^(2*j)
//        
/// Provides _stochastic_ Gradient Descent optimization
/// as described in Lei-Jordan On the adaptativity of stochastic gradient based optimisation 2019
pub struct StochasticControlledGradientDescent {
    rng: Xoshiro256PlusPlus,
    // parameter governing evolution of inner_loop_size
    m_zero: f64,
    // m_0 in the paper
    mini_batch_size_init : usize,
    // B_0 in Paper
    large_batch_size_init: usize,
    //
    batch_growing_factor: f64,
}

impl  StochasticControlledGradientDescent {
    pub fn new(m_zero: f64, mini_batch_size_init : usize, large_batch_size_init: usize, batch_growing_factor : f64) -> StochasticControlledGradientDescent {
        //
        trace!(" mini batch size {:?} , large_batch_size_init {:?}, batch_growing_factor {:2.4E}", mini_batch_size_init, large_batch_size_init, batch_growing_factor);
        //
        StochasticControlledGradientDescent {
            rng : Xoshiro256PlusPlus::seed_from_u64(4664397),
            m_zero : m_zero,
            mini_batch_size_init : mini_batch_size_init,
            large_batch_size_init : large_batch_size_init,
            // alfa in the paper.
            batch_growing_factor : batch_growing_factor,
        }
    }
    /// Seeds the random number generator using the supplied `seed`.
    /// This is useful to create re-producable results.
    pub fn seed(&mut self, seed: [u8; 32]) {
        self.rng = Xoshiro256PlusPlus::from_seed(seed);
    }

    // batch growing factor cannot be too large, so it must be adjusted accoding to nbterms value, and nb_max_iterations
    // We use the following rules.
    // For large batch sizes:
    //  1.   B_0 max(10, nbterms/100)
    //  2.   B_0 * alfa^(2T) < nbterms
    // For the number of mini batch, we need to avoid large variances so 
    //  1.  
    fn estimate_batch_growing_factor(&self, nb_max_iterations : usize , nbterms:usize) -> f64 {
        let batch_growing_factor : f64;
        if self.m_zero > nbterms as f64 {
            warn!("m_zero > nbterms in functio to minimize, exiting");
            std::process::exit(1);
        }
        //
        let log_alfa = ((nbterms as f64).ln() -self.m_zero.ln()) / (2. * nb_max_iterations as f64);
        batch_growing_factor = log_alfa.exp();
        trace!(" upper bound for batch_growing_factor  {:?}",  batch_growing_factor);
        //
        return batch_growing_factor;
    } // end of estimate_batch_growing_factor



    ///
    pub fn get_batch_size_at_jstep(&self, batch_growing_factor : f64, nbterms : usize, j: usize) -> BatchSizeInfo {
        let alfa_j = batch_growing_factor.powi(j as i32);
        BatchSizeInfo {
            _step : j,
            large_batch : (self.large_batch_size_init * (alfa_j * alfa_j).ceil() as usize).min(nbterms),
            mini_batch : (((self.mini_batch_size_init as f64) * alfa_j).floor() as usize).min(nbterms),
            nb_mini_batch_parameter : self.m_zero * alfa_j,
        }
    } // end of get_batch_size_at_jstep
    ///
    fn get_step_size_at_jstep(&self, j:usize) -> f64 {
        let step_size = 1./(1+j) as f64;
        step_size.sqrt()
    }
    /// 
    /// sample number of mini batch according to geometric law of parameter p = b_j/(m_j+b_j) 
    /// with law : P(N=k) = (1-p) * p^k. (proba of nb trial before success mode)
    fn sample_nb_small_mini_batches(&self, batch_size_info : &BatchSizeInfo,
                rng : &mut Xoshiro256PlusPlus) -> usize {
        let mut n_j = 1;
        let m_j =  batch_size_info.nb_mini_batch_parameter as f64;
        let b_j = batch_size_info.mini_batch as f64;
        let p : f64 = m_j/(m_j+b_j);
        trace!(" geometric law parameter {:2.4E}", p);
        // pass a &mut here!!
        let mut xsi : f64 = rand_distr::Standard.sample(rng);
        // success is when xsi is >= p !!
        while xsi < p {
            xsi = rand_distr::Standard.sample(rng);
            n_j += 1;
        }
        trace!(" nb small mini batch {:?} m_j {:2.4E} b_j : {:2.4E} ", n_j,  m_j, b_j);
        return n_j;
    }
    
} // end impl StochasticControlledGradientDescent


#[allow(dead_code)]
// if size_asked > size_in all terms are accepted, we get a full gradient!
fn sample_without_replacement_from_slice(size_asked: usize, in_terms: &[usize], rng : &mut Xoshiro256PlusPlus) -> Vec<usize> {
        let mut out_terms = Vec::<usize>::with_capacity(size_asked.min(in_terms.len()));
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
}  // end of sample_without_replacement_from_slice





// this function requires that size_in be equal to in_temrs.count() !!!!
// but it can be useful as it enables call with a range and thus avoid passing reference to large slice!
// if size_asked > size_in all terms are accepted, we get a full gradient!
fn sample_without_replacement_iter(size_asked: usize, in_terms: impl IntoIterator<Item=usize>, size_in : usize, rng : &mut Xoshiro256PlusPlus) -> Vec<usize> {
    let mut out_terms = Vec::<usize>::with_capacity(size_asked.min(size_in));
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

    fn minimize(&self, function: &F, initial_position: Vec<f64>, nb_max_iterations : usize) -> Solution {
        let mut position = initial_position;
        let dimension = position.len();
        let mut value = function.value(&position);
        let mut direction : Vec<f64> = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            direction.push(0.);
        } 

        if log_enabled!(Info) {
            info!("Starting with y = {:e} for x = {:?}", value, position);
        } else {
            info!("Starting with y = {:e}", value);
        }

        let mut iteration : usize = 0;
        let mut rng = self.rng.clone();
        let nb_terms = function.terms();
        trace!("nb_max_iterations {:?}", nb_max_iterations);
        let batch_growing_factor = self.estimate_batch_growing_factor(nb_max_iterations, function.terms());

        loop {
            // get iteration parameters
            let iter_params = self.get_batch_size_at_jstep(batch_growing_factor, nb_terms, iteration);
            // sample large batch of size Bj
            let large_batch_indexes = sample_without_replacement_iter(iter_params.large_batch, 0..nb_terms, nb_terms, & mut rng);
            trace!("\n iter {:?} got large batch size {:?}, mini batch param {:2.4E}", 
                    iteration, large_batch_indexes.len(), iter_params.nb_mini_batch_parameter);
            // compute gradient on large batch index set and store initial position
            let large_batch_gradient = function.partial_gradient(&position, &large_batch_indexes);
            let position_before_mini_batch = position.clone();
            let mut position_during_mini_batches = position.clone();
            // sample binomial law for number Nj of small batch iterations
            let n_j = self.sample_nb_small_mini_batches(&iter_params, &mut rng);
            // loop on small batch iterations
            for _k in 0..n_j {
                // sample mini batch terms
                let terms = sample_without_replacement_iter(iter_params.mini_batch, 0..nb_terms, nb_terms, &mut rng);
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
                // compute gradient norm
                let mut norm = direction.iter().fold(0., | acc, x  | acc + x*x);
                norm = norm.sqrt();
                debug!("Iteration {:?} y = {:2.4E}, | direction| = {:2.4e}", iteration, value, norm);
            } else {
                debug!("Iteration {:?}  y = {:2.4E}", iteration, value);
            }
            // convergence control or max iterations control
            if iteration >= nb_max_iterations {
                    info!("Reached maximal number of iterations required , stopping optimization");
                    return Solution::new(position, value);
            }
        } // end global loop

    } // end minimize
}  // end impl impl<F: Summation1> Minimizer<F>



