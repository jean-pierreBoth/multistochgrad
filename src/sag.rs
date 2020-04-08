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
        // some logs
        if log_enabled!(Info) {
            info!("Starting with y = {:2.4e}", value);
        } else {
            info!("Starting with y = {:e}", value);
        }
        trace!("nb_max_iterations {:?}", nb_max_iterations);
        // some temporaries
        let mut term_gradient_current : Array<f64, D>;
        term_gradient_current = position.clone();
        term_gradient_current.fill(0.);        
        // 
        let mut rng = self.rng.clone();
        let nb_terms = function.terms();
        let mut monitoring = Vec::<IterRes>::with_capacity(nb_terms);
        //
        let mut iteration = 0;
        //
        let mut direction = initial_position.clone();
        direction.fill(0.);        

        let mut terms_seen = Vec::<u8>::with_capacity(function.terms());
        let mut nb_terms_seen = 0;
        let mut gradient_list = Vec::<Box<Array<f64,D>>>::with_capacity(function.terms());
        for _term in 0..nb_terms {
//            function.partial_gradient(&position, &[_term], &mut term_gradient_current);
            gradient_list.push(Box::new(term_gradient_current.clone()));
            terms_seen.push(0);
        }

        //
        loop {
            // sample unbiaised ...) i in 0..nbterms
            let xsi : f64 = rand_distr::Standard.sample(&mut rng);
            let term = (nb_terms as f64 * xsi).floor() as usize;
            if terms_seen[term] == 0 {
                terms_seen[term] = 1;
                nb_terms_seen += 1;
            }
            //  trace!(" term_gradient_current {:2.4E} ", &crate::types::norm_l2(&term_gradient_current));
            // gradient terms do not have the nb_terms renormalization
            function.partial_gradient(&position, &[term], &mut term_gradient_current);
    //        trace!(" term {:?} gradient {:2.6E}  nb_term {:?} ", term, &term_gradient_current, nb_terms_seen);
    //        trace!(" term {:?} gradient substracted  {:2.6E} ", term, gradient_list[term].as_ref());
            direction = direction - (gradient_list[term].as_ref() - &term_gradient_current) / (nb_terms as f64);
            gradient_list[term] = Box::new(term_gradient_current.clone());
       
            position = position - self.get_step_size_at_jstep(iteration) * &direction;
            iteration += 1;
            // monitoring
            value = function.value(&position);
            let gradnorm = crate::types::norm_l2(&direction);
            monitoring.push(IterRes {
                value : value,
                gradnorm : gradnorm,
            });
            if log_enabled!(Debug) && iteration % 100 == 0 {
                trace!(" position {:2.6E} ", &position);
                trace!(" direction {:2.6E} ", &direction);
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
