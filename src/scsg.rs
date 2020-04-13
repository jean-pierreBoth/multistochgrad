//! A Rust implementation of Lei-Jordan paper 
//! On the adaptativity of Stochastic gradient based optiimization
//! https://arxiv.org/abs/1904.04480

use log::Level::*;
use log::{debug, info, warn, trace, log_enabled};

use rand::{SeedableRng};
use rand::distributions::{Distribution};
// a fast but non crypto secure algo. method jump to use in // !!!!
use rand_xoshiro::Xoshiro256PlusPlus;
use ndarray::{Array, Dimension};

use crate::types::*;
use crate::monitor::*;



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
    // step size
    step_size : f64,
}

/// Provides Stochastic Controlled Gradient Descent optimization
/// as described in Lei-Jordan On the adaptativity of stochastic gradient based optimisation 2019
/// 
/// One iteration j consists in a large batch of size Bⱼ and then a number noted mⱼ of small batches
/// of size bⱼ and update position with a step ηⱼ
/// 
/// The papers establishes rates of convergence depending on the ratio 
/// mⱼ/Bⱼ , bⱼ/mⱼ and ηⱼ/bⱼ and their products.
///  
/// If nbterms is the number of terms in function to minimize and j the iteration number:
/// 
///       Bⱼ evolves as :   large_batch_size_init * nbterms * alfa^(2j)
///       mⱼ evolves as :   m_zero *  nbterms * alfa^(3j/2)
///       bⱼ evolves as :   b_0 * alfa^j
///       ηⱼ evolves as :   eta_0 / alfa^j
///     
///     where alfa is computed to be slightly greater than 1.  
///     In fact α is chosen so that :  B_0 * alfa^(2*nbiter) = nbterms
/// 
///  The evolution of Bⱼ is bounded above by nbterms/10 and bⱼ by nbterms/100.  
///  The size of small batch must stay small so b₀ must be small (typically 1 seems OK)
///  
/// 
pub struct StochasticControlledGradientDescent {
    rng: Xoshiro256PlusPlus,
    /// step_size initialization
    eta_zero : f64,
    /// fraction of nbterms to consider in initialization of mⱼ governing evolution of nb small iterations
    m_zero: f64,
    /// m_0 in the paper
    mini_batch_size_init : usize,
    /// related to B_0 in Paper. Fraction of terms to consider in initialisation of B_0
    large_batch_size_init: f64,
}

impl  StochasticControlledGradientDescent {
    /// args are :
    ///   - initial value of step along gradient value of 0.5 is a good default choice.
    ///   - fraction of nbterms to initialize large batch size : a good default value is around 0.01 so that  
    ///              large batch size begins at 0.01 * nbterms
    ///   - m_zero : a good value is 0.1 so that times large_batch_size_init so mⱼ << Bⱼ
    ///   - base value for size of mini_batchs : a value of 1 is a good default choice
    /// 
    pub fn new(eta_zero : f64, m_zero: f64, mini_batch_size_init : usize, large_batch_size_init: f64) -> StochasticControlledGradientDescent {
        //
        if large_batch_size_init >  1. {
            warn!("large_batch_size_init > 1. , fraction factor for large_batch size initialization must be < 1. , exiting");
            std::process::exit(1);
        }
        if m_zero  > large_batch_size_init {
            warn!("m_zero > large_batch_size_init fraction , base fraction for nb mini batch should be less than for large_batch_size");
            std::process::exit(1);
        }
        info!(" eta_zero {:2.4E} m_zero {:2.4} \n mini batch size {:?} , large_batch_size_init {:2.4E}", 
            eta_zero, m_zero, mini_batch_size_init, large_batch_size_init);
        //
        StochasticControlledGradientDescent {
            rng : Xoshiro256PlusPlus::seed_from_u64(4664397),
            eta_zero : eta_zero,
            m_zero : m_zero,
            mini_batch_size_init : mini_batch_size_init,
            large_batch_size_init : large_batch_size_init as f64,
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
    fn estimate_batch_growing_factor(&self, nb_max_iterations : usize , nbterms:usize) -> f64 {
        let batch_growing_factor : f64;
        if self.m_zero > nbterms as f64 {
            warn!("m_zero > nbterms in functio to minimize, exiting");
            std::process::exit(1);
        }
        //
        let log_alfa = (-self.large_batch_size_init.ln()) / (2. * nb_max_iterations as f64);
        batch_growing_factor = log_alfa.exp();
        if batch_growing_factor <= 1. {
            println!("batch growing factor shoud be greater than 1. , possibly you can reduce number of iterations ");
        }
        debug!(" upper bound for batch_growing_factor  {:2.4E}",  batch_growing_factor);
        //
        return batch_growing_factor;
    } // end of estimate_batch_growing_factor



    // returns BatchSizeInfo for current iteration
    fn get_batch_size_at_jstep(&self, batch_growing_factor : f64, nbterms : usize, j: usize) -> BatchSizeInfo {
        let alfa_j = batch_growing_factor.powi(j as i32);
        // max size of large batch is 100 or 0.1 * the number of terms
        let max_large_batch_size;
        if nbterms > 100 {
            max_large_batch_size = (nbterms as f64/10.).ceil() as usize;
        }
        else {
            max_large_batch_size = nbterms;
        }
        // ensure max_mini_batch_size is at least 1.
        let max_mini_batch_size = (nbterms as f64/100.).ceil() as usize;
        // B_j
        let large_batch_size = ((self.large_batch_size_init * (nbterms as f64) * alfa_j * alfa_j).ceil() as usize).min(max_large_batch_size);
        // b_j  grow slowly as log(1. + )
        let mini_batch_size = ((self.mini_batch_size_init as f64 * alfa_j).floor() as usize).min(max_mini_batch_size);
        // m_j  computed to ensure mean number of mini batch < large_batch_size as mini_batch_size_init < large_batch_size_init is enfored
        let nb_mini_batch_parameter = self.m_zero * (nbterms as f64) * alfa_j.powf(1.5);
        let step_size = self.eta_zero / alfa_j;
        //
        BatchSizeInfo {
            _step : j,
            large_batch : large_batch_size,
            mini_batch : mini_batch_size,
            nb_mini_batch_parameter : nb_mini_batch_parameter,
            step_size : step_size,
        }
    } // end of get_batch_size_at_jstep
    /// 
    /// sample number of mini batch according to geometric law of parameter p = b_j/(m_j+b_j) 
    /// with law : P(N=k) = (1-p) * p^k. (proba of nb trial before success mode)
    fn get_nb_small_mini_batches(&self, batch_size_info : &BatchSizeInfo) -> usize {
        let m_j =  batch_size_info.nb_mini_batch_parameter as f64;
        let b_j = batch_size_info.mini_batch as f64;
        let p : f64 = m_j/(m_j+b_j);
        trace!(" geometric law parameter {:2.4E}", p);
        // we return mean of geometric. Sampling too much instable due to large variance of geometric distribution.
        let mut n_j = (p / (1.-p)).ceil() as usize;
        n_j = n_j.min(batch_size_info.large_batch);
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
    assert_eq!(size_asked, out_terms.len());
    out_terms 
}  // end of sample_without_replacement



impl<D:Dimension, F: SummationC1<D>> Minimizer<D, F, usize> for  StochasticControlledGradientDescent {
    type Solution = Solution<D>;

    fn minimize(&self, function: &F, initial_position: &Array<f64,D>, max_iterations : Option<usize>) -> Solution<D> {
        let mut position = initial_position.clone();
        let mut value = function.value(&position);
        let nb_max_iterations = max_iterations.unwrap();
        // direction propagation
        let mut direction : Array<f64, D> = position.clone();
        direction.fill(0.);

        if log_enabled!(Info) {
            info!("Starting with y = {:e} \n for x = {:?}", value, position);
        } else {
            info!("Starting with y = {:e}", value);
        }
        trace!("nb_max_iterations {:?}", nb_max_iterations);
        //
        let mut iteration : usize = 0;
        let mut rng = self.rng.clone();
        let nb_terms = function.terms();
        let mut monitoring = IterationRes::<D>::new(nb_max_iterations, SolMode::Last);
        let batch_growing_factor = self.estimate_batch_growing_factor(nb_max_iterations, function.terms());
        // temporary gradients passed by ref to avoid possibly large reallocation
        let mut large_batch_gradient: Array<f64, D> = position.clone();
        large_batch_gradient.fill(0.);
        //
        let mut  mini_batch_gradient_current : Array<f64, D>;
        mini_batch_gradient_current = position.clone();
        mini_batch_gradient_current.fill(0.);
        //
        let mut  mini_batch_gradient_origin : Array<f64, D>;
        mini_batch_gradient_origin = position.clone();
        mini_batch_gradient_origin.fill(0.);  
        // now we work      
        loop {
            // get iteration parameters
            let iter_params = self.get_batch_size_at_jstep(batch_growing_factor, nb_terms, iteration);
            let step_size = iter_params.step_size;
            // sample large batch of size Bj
            let large_batch_indexes = sample_without_replacement_iter(iter_params.large_batch, 0..nb_terms, nb_terms, & mut rng);
            trace!("\n iter {:?} got large batch size {:?}, mini batch param {:2.4E}, mini batch size {:?}, step {:2.4E}", 
                    iteration, large_batch_indexes.len(), iter_params.nb_mini_batch_parameter, 
                    iter_params.mini_batch, iter_params.step_size);
            // compute gradient on large batch index set and store initial position
            function.mean_partial_gradient(&position, &large_batch_indexes, &mut large_batch_gradient);
            let position_before_mini_batch = position.clone();
            let mut position_during_mini_batches = position.clone();
            // sample binomial law for number Nj of small batch iterations
            let n_j = self.get_nb_small_mini_batches(&iter_params);
            // loop on small batch iterations
            for _k in 0..n_j {
                // sample mini batch terms
                let terms = sample_without_replacement_iter(iter_params.mini_batch, 0..nb_terms, nb_terms, &mut rng);
                //
                function.mean_partial_gradient(&position_during_mini_batches, &terms, &mut mini_batch_gradient_current);
                //
                function.mean_partial_gradient(&position_before_mini_batch, &terms, &mut mini_batch_gradient_origin);
                //
                // if log_enabled!(Trace)  {
                //     if _k == 0 {
                //         assert!(norm_l2(&mini_batch_gradient_origin) > 0.);
                //         trace!("mini_batch_gradient_origin  L2 {:2.4E} ", norm_l2(&mini_batch_gradient_origin));
                //     }
                //     else {
                //         trace!("mini_batch_gradient_current L2 {:2.4E} ", norm_l2(&mini_batch_gradient_current));
                //         assert!(norm_l2(&mini_batch_gradient_current) > 0.);
                //     }
                // }
                //
                direction = &mini_batch_gradient_current - &mini_batch_gradient_origin + &large_batch_gradient;
                // step into the direction of the negative gradient
                position_during_mini_batches = position_during_mini_batches - step_size * &direction;
            } // end mini batch loop
            // update position
            position = position_during_mini_batches.clone();
            iteration += 1;
            // some monitoring
            value = function.value(&position);
            let gradnorm = norm_l2(&direction);
            monitoring.push(value, &position, gradnorm);
            if log_enabled!(Debug) {
                trace!(" direction {:2.6E} ", gradnorm);
                debug!("\n\n Iteration {:?} y = {:2.4E}", iteration, value);
            }
            // convergence control or max iterations control
            if iteration >= nb_max_iterations {
                info!("Reached maximal number of iterations required , stopping optimization");
                let rank = monitoring.check_monoticity();
                println!("monotonous convergence from rank : {:?}", rank);

                return Solution::new(position, value);
            }
        } // end global loop

    } // end minimize
}  // end impl impl<F: Summation1> Minimizer<F>


