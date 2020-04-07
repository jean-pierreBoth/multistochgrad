//! Implementation of basic Stochastic Average Gradient (SAG)
//! 
//! 
//! 
//!  
//


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


pub struct SagDescent {
    //
    rng: Xoshiro256PlusPlus,
    /// step size
    step_size : f64,
    ///
    _momentum : Option<f64>,
}



impl SagDescent {
    pub fn new(step_size : f64) -> SagDescent {
        //
        trace!(" step_size {:2.4E} ", step_size);
        //
        SagDescent {
            rng : Xoshiro256PlusPlus::seed_from_u64(4664397),
            step_size : step_size,
            _momentum : None,
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
} // end impl SagDescent



impl<D:Dimension, F: SummationC1<D>> Minimizer<D, F> for  SagDescent {
    type Solution = Solution<D>;

    fn minimize(&self, function: &F, initial_position: &Array<f64,D>, nb_max_iterations : usize) -> Solution<D> {
        let mut position = initial_position.clone();
        let mut value = function.value(&position);

        if log_enabled!(Info) {
            info!("Starting with y = {:2.4e}", value);
        } else {
            info!("Starting with y = {:e}", value);
        }
        trace!("nb_max_iterations {:?}", nb_max_iterations);
        // direction propagation
        let mut direction : Array<f64, D> = position.clone();
        direction.fill(0.);
        let mut term_gradient_old : Array<f64, D>;
        term_gradient_old = position.clone();
        term_gradient_old.fill(0.);
        let mut term_gradient_current : Array<f64, D>;
        term_gradient_current = position.clone();
        term_gradient_current.fill(0.);        
        // 
        let mut rng = self.rng.clone();
        let nb_terms = function.terms();
        let mut monitoring = Vec::<IterRes>::with_capacity(nb_terms);
        
        let mut iteration = 0;
        //
        loop {
            // sample unbiaised ...) i in 0..nbterms
            let xsi : f64 = rand_distr::Standard.sample(&mut rng);
            let term = (nb_terms as f64 * xsi.floor()) as usize;
            function.partial_gradient(&position, &[term], &mut term_gradient_current);
            direction = direction - &term_gradient_old + &term_gradient_current;
            position = position - self.get_step_size_at_jstep(iteration) * &direction;
            iteration += 1;

            value = function.value(&position);
            let gradnorm = crate::types::norm_l2(&direction);
            monitoring.push(IterRes {
                value : value,
                gradnorm : gradnorm,
            });
            if log_enabled!(Debug) {
                trace!(" direction {:2.6E} ", &gradnorm);
                debug!("\n\n Iteration {:?} y = {:2.4E}", iteration, value);
            }
            // convergence control or max iterations control
            if iteration >= nb_max_iterations {
                info!("Reached maximal number of iterations required , stopping optimization");
                return Solution::new(position, value);
            }
        }    
    } // end of minimize

} // end of impl Minimizer
