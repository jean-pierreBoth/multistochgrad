//! Implementation of basic Stochastic Average Gradient (SAG)
//! 
//! The Stochastic Averaged Gradient Descent as described in the paper:
//! "Minimizing Finite Sums with the Stochastic Average Gradient" (2013, 2016)
//! M.Schmidt, N.LeRoux, F.Bach
//! 
//!  
//!


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


///
/// Provides stochastic Average Gradient Descent optimization described in : 
///  *"Minimizing Finite Sums with the Stochastic Average Gradient" (2013, 2016)*
///   M.Schmidt, N.LeRoux, F.Bach
/// 
/// We implemented a mini batch version as it consumes less memory. Moreover
/// we found it is more stable.
/// 
/// The algorithm never computes a full gradient. It initialize the direction of propagation
/// as a null vector. 
/// The terms of the sum as grouped in blocks. At each iteration
/// the gradient of randomly selected block is computed at the current position added
/// to the current total gradient from which the gradient of the selected block at
/// the last iteration it was selected. So the current estimate of the gradient 
/// is made of blocks contributions that are more or less oudated.
/// 
/// It must be noted that a gradients of each block are stored and updated for future use when
/// the block is reused.
/// The larger the block size, the lesser the number of blocks but more work is done at each iteration.
/// 
/// 
/// Precisely we have the following sequence:  
/// - random selection of a block  
/// - computation of the partial gradient of the sum of terms in the block 
/// - computation of direction of propagation as the batch gradient + gradient of block at current position 
///                     - gradient of the same block last time it was computed.
/// - update of position
/// 
/// The step size is constant (and could automatically adjusted to Lipschitz constant in a later work)
/// 
pub struct SagDescent {
    //
    rng: Xoshiro256PlusPlus,
    /// batch size for partial update of gradient
    batch_size : usize,
    /// step size
    step_size : f64,
    ///
    _momentum : Option<f64>,
}



impl SagDescent {
    /// size of batch or block to use.
    pub fn new(batch_size: usize, step_size : f64) -> SagDescent {
        //
        trace!(" batch size {:?} step_size {:2.4E} ", batch_size, step_size);
        //
        SagDescent {
            rng : Xoshiro256PlusPlus::seed_from_u64(4664397),
            batch_size : batch_size,
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



impl<D:Dimension, F: SummationC1<D>> Minimizer<D, F, usize> for  SagDescent {
    type Solution = Solution<D>;

    fn minimize(&self, function: &F, initial_position: &Array<f64,D>, max_iterations : Option<usize>) -> Solution<D> {
        let mut position = initial_position.clone();
        let mut value = function.value(&position);
        let nb_max_iterations = max_iterations.unwrap();
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
        let mut monitoring = IterationRes::<D>::new(nb_max_iterations, SolMode::Last);
        //
        let mut iteration = 0;
        //
        let mut direction = initial_position.clone();
        direction.fill(0.);        
        // blocks 
        let mut nb_block_seen = 0;
        let block_size = self.batch_size;
        let nb_blocks = if nb_terms % block_size == 0 {
                nb_terms / block_size
        }
        else {
            nb_terms / block_size + 1
        };
        let mut block_seen = Vec::<u8>::with_capacity(nb_blocks);
        let mut gradient_list = Vec::<Box<Array<f64,D>>>::with_capacity(nb_blocks);
        for _block in 0..nb_blocks {
            gradient_list.push(Box::new(term_gradient_current.clone()));
            block_seen.push(0);
        }
        let block_start = | i : usize | -> usize {
            let start = i * block_size;
            start
        };
        let block_end =  | i : usize | -> usize {
            let end = ((i+1) * block_size).min(nb_terms);
            end
        };
        //
        loop {
            // sample unbiaised ...) i in 0..nbterms
            let xsi : f64 = rand_distr::Standard.sample(&mut rng);
            let block = (nb_blocks as f64 * xsi).floor() as usize;
            if block_seen[block] == 0 {
                block_seen[block] = 1;
                nb_block_seen += 1;
                if nb_block_seen >= nb_blocks {
                    info!(" all blocks were visited at iter {:?} ", iteration);
                }
            }
            // trace!(" term_gradient_current {:2.4E} ", &crate::types::norm_l2(&term_gradient_current));
            let block_content = &(block_start(block) .. block_end(block)).into_iter().collect::<Vec<usize>>();
            function.partial_gradient(&position, block_content, &mut term_gradient_current);
    //        trace!(" term {:?} gradient {:2.6E}  nb_term {:?} ", term, &term_gradient_current, nb_terms_seen);
    //        trace!(" term {:?} gradient substracted  {:2.6E} ", term, gradient_list[term].as_ref());
            direction = direction - (gradient_list[block].as_ref() - &term_gradient_current) / (nb_terms as f64);
            gradient_list[block] = Box::new(term_gradient_current.clone());
            // update position
            position = position - self.get_step_size_at_jstep(iteration) * &direction;
            iteration += 1;
            // monitoring
            value = function.value(&position);
            let gradnorm = crate::types::norm_l2(&direction);
            monitoring.push(value, &position, gradnorm);
            if log_enabled!(Debug) && (iteration % nb_terms == 0) ||  iteration % 10 == 0 {
//                trace!(" position {:2.6E} ", &position);
//                trace!(" direction {:2.6E} ", &direction);
                debug!("\n\n Iteration {:?} y = {:2.4E}, norm grad {:?}", iteration, value, gradnorm);
            }
            // convergence control or max iterations control
            if iteration % nb_terms == 0 && gradnorm < 1.0e-5 {
                info!("Null gradient , stopping optimization {:2.4E}", gradnorm);
                return Solution::new(position, value);
            }
            if iteration >= nb_max_iterations {
                info!("Reached maximal number of iterations required , stopping optimization");
                let rank = monitoring.check_monoticity();
                info!(" monotonous convergence from rank : {:?}", rank);
                return Solution::new(position, value);
            }
        }    
    } // end of minimize

} // end of impl Minimizer
