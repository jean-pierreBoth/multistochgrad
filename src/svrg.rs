//! A Rust implementation of Johnsohn-Zhang paper 
//! Acceleration stochastic Gradient Descent using Predictive Variance Reduction
//! https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf


#[allow(unused_imports)]
use log::Level::*;
#[allow(unused_imports)]
use log::{debug, info, warn, trace, log_enabled};

use rand::{SeedableRng};
use rand::distributions::{Distribution};
// a fast but non crypto secure algo. method jump to use in // !!!!
use rand_xoshiro::Xoshiro256PlusPlus;
use ndarray::{Array, Dimension};

use crate::types::*;
use crate::monitor::*;



// m_zero should be around 2 * mini_batch_size_init 
// we try mini batch size m_zero * alfa^j 
//        large batch size large_batch_size_init * alfa^(2*j)
//        
/// Provides _stochastic_ Gradient Descent optimization
/// as described in Lei-Jordan On the adaptativity of stochastic gradient based optimisation 2019
pub struct SVRGDescent {
    rng: Xoshiro256PlusPlus,
    // parameter governing evolution of inner_loop_size
    nb_mini_batch: usize,
    // m_0 in the paper
    step_size : f64,
}

impl SVRGDescent  {
    pub fn new(nb_mini_batch : usize , step_size : f64) -> SVRGDescent {
        //
        trace!(" nb_mini_batch {:?} step_size {:2.4E} ", nb_mini_batch, step_size);
        //
        SVRGDescent {
            rng : Xoshiro256PlusPlus::seed_from_u64(4664397),
            nb_mini_batch : nb_mini_batch,
            step_size : step_size,
        }
    }
    /// Seeds the random number generator using the supplied `seed`.
    /// This is useful to create re-producable results.
    pub fn seed(&mut self, seed: [u8; 32]) {
        self.rng = Xoshiro256PlusPlus::from_seed(seed);
    }
    ///
    fn get_step_size_at_jstep(&self, _j:usize) -> f64 {
        self.step_size
    }
    ///
    fn get_nb_small_mini_batches(&self, _j:usize) -> usize {
        self.nb_mini_batch
    }
    
} // end impl SVRGDescent


impl<D:Dimension, F: SummationC1<D>> Minimizer<D, F> for  SVRGDescent {
    type Solution = Solution<D>;

    fn minimize(&self, function: &F, initial_position: &Array<f64,D>, nb_max_iterations : usize) -> Solution<D> {
        let mut position = initial_position.clone();
        let mut value = function.value(&position);
        // direction propagation
        let mut direction : Array<f64, D> = position.clone();
        direction.fill(0.);

        if log_enabled!(Info) {
            info!("Starting with y = {:e} for x = {:?}", value, position);
        } else {
            info!("Starting with y = {:e}", value);
        }
        trace!("nb_max_iterations {:?}", nb_max_iterations);

        let mut iteration : usize = 0;
        let nb_terms = function.terms();
        let mut monitoring = Vec::<IterRes>::with_capacity(nb_max_iterations);
        //
        let mut  term_gradient_current : Array<f64, D>;
        term_gradient_current = position.clone();
        term_gradient_current.fill(0.);
        //
        let mut term_gradient_origin : Array<f64, D>;
        term_gradient_origin = position.clone();
        term_gradient_origin.fill(0.);  
        // now we work      
        let mut rng = self.rng.clone();
        loop {
            // get iteration parameters
            let batch_gradient = function.gradient(&position);
            let position_before_mini_batch = position.clone();
            let mut position_during_mini_batches = position.clone();
            // sample binomial law for number Nj of small batch iterations
            let n_j = self.get_nb_small_mini_batches(iteration);
            // loop on small batch iterations
            for _k in 0..n_j {
                // sample mini batch terms
                // 
                let xsi : f64 = rand_distr::Standard.sample(&mut rng);
                let term = (nb_terms as f64 * xsi.floor()) as usize;
                function.partial_gradient(&position_during_mini_batches, &[term], &mut term_gradient_current);
                //
                function.partial_gradient(&position_before_mini_batch, &[term], &mut term_gradient_origin);
                //
                // if log_enabled!(Trace)  {
                //     if _k == 0 {
                //         assert!(norm_l2(&term_gradient_origin) > 0.);
                //         trace!("term_gradient_origin  L2 {:2.4E} ", norm_l2(&term_gradient_origin));
                //     }
                //     else {
                //         trace!("mini_batch_gradient_current L2 {:2.4E} ", norm_l2(&term_gradient_current));
                //         assert!(norm_l2(&mini_batch_gradient_current) > 0.);
                //     }
                // }
                //
                direction = &term_gradient_current - &term_gradient_origin + &batch_gradient;
                // step into the direction of the negative gradient
                position_during_mini_batches = position_during_mini_batches - self.get_step_size_at_jstep(iteration) * &direction;
            } // end mini batch loop
            // update position
            position = position_during_mini_batches.clone();
            iteration += 1;

            value = function.value(&position);
            let gradnorm = norm_l2(&direction);
            monitoring.push(IterRes {
                value : value,
                gradnorm : gradnorm,
            });
            //
            if log_enabled!(Debug) {
                trace!(" direction {:2.6E} ", &gradnorm);
                debug!("\n\n Iteration {:?} y = {:2.4E}", iteration, value);
            }
            // convergence control or max iterations control
            if iteration >= nb_max_iterations {
                info!("Reached maximal number of iterations required , stopping optimization");
                return Solution::new(position, value);
            }
        } // end global loop

    } // end minimize
}  // end impl impl<F: Summation1> Minimizer<F>



