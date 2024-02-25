//! A Rust implementation of Lei-Jordan papers:     
//! "On the adaptativity of Stochastic gradient based optimization" (2019) [SCSG-1](https://arxiv.org/abs/1904.04480)  
//! "Less than a single pass : stochastically controlled stochastic gradient" (2019)  [SCSG-2](https://arxiv.org/abs/1609.03261)

use log::Level::*;
use log::{debug, info, log_enabled, trace, warn};

use cpu_time::ProcessTime;
use std::time::SystemTime;

use std::iter::FromIterator;

use rand::distributions::Distribution;
use rand::SeedableRng;
// a fast but non crypto secure algo. method jump to use in // !!!!
use ndarray::{Array, Dimension};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::monitor::*;
use crate::types::*;

// a struct to have batch size (large and mini) to use at a given iteration
// we must have large_batch >= nb_mini_batch_parameter >= mini_batch  (Cf Th 4.1)
pub struct BatchSizeInfo {
    _step: usize,
    // large batch size
    large_batch: usize,
    // mini batvh size
    mini_batch: usize,
    // nb mini bath parameter
    nb_mini_batch_parameter: f64,
    // step size
    step_size: f64,
}

///
/// Provides Stochastic Controlled Gradient Descent optimization based on 2 papers of Lei-Jordan.  
/// - ``"On the adaptativity of stochastic gradient based optimisation"`` arxiv 2019  [SCSG-1](https://arxiv.org/abs/1904.04480)
/// - ``"Less than a single pass : stochastically controlled stochastic gradient"`` arxiv 2019 [SCSG-2](https://arxiv.org/abs/1609.03261)
///
/// According to the first paper we have the following notations:
///  
/// One iteration j consists in :
///    - a large batch of size Bⱼ
///    - a number noted mⱼ of small batches of size bⱼ
///    - update position with a step ηⱼ.
/// The number of mini batch is described by a random variable with a geometric law.
///
/// The paper establishes rates of convergence depending on the ratio
/// mⱼ/Bⱼ , bⱼ/mⱼ and ηⱼ/bⱼ and their products.
///  
/// The second paper :  
/// "Less than a single pass : stochastically controlled stochastic gradient"  
/// describes a simplified version where the mini batches consist in just one term
/// and the number of mini batch is set to the mean of the geometric variable corresponding to
/// number of mini batches.
///   
/// We adopt a mix of the two papers:
/// It seems that letting the size of mini batch grow a little is more stable than keeping it to 1.
/// (in particular when initialization of the algorithm varies.)
/// but replacing the geometric law by its mean is really more stable due to the large variance of its law.
///
/// We choose a fraction of the number of terms in the sum (*large_batch_fraction_init*) and $alfa$
/// so that large_batch_fraction_init * alfa^(2*nbiter) = 1.  
///
/// Then :  
/// If nbterms is the number of terms in function to minimize and j the iteration number:
///
///  -    Bⱼ evolves as :   large_batch_fraction_init * nbterms * alfa^(2j)
///  -    mⱼ evolves as :   m_zero * nbterms * alfa^(3j/2)
///  -    bⱼ evolves as :   b_0 * alfa^j
///  -    ηⱼ evolves as :   eta_0 / alfa^(j/2)
///     
///
///  The evolution of Bⱼ is bounded above by nbterms/10 and bⱼ by nbterms/100.  
///  The size of small batch must stay small so b₀ must be small (typically 1 seems OK)
///  
///
pub struct StochasticControlledGradientDescent {
    rng: Xoshiro256PlusPlus,
    /// step_size initialization
    eta_zero: f64,
    /// multiplicative factor in mⱼ,  governing evolution of nb small iterations
    m_zero: f64,
    /// b_0 in the paper. governs the size of mini batches
    mini_batch_size_init: usize,
    /// related to B_0 in Paper. Fraction of terms to consider in initialisation of B_0
    large_batch_fraction_init: f64,
}

impl StochasticControlledGradientDescent {
    /// args are :
    ///   - eta_zero : initial value of step along gradient value of 0.5 is a good default choice.
    ///   - m_zero : a good value is 0.2 *large_batch_fraction_init so that  mⱼ << Bⱼ
    ///   - mini_batch_size_init : base value for size of mini_batchs : a value of 1 is a good default choice
    ///   - large_batch_fraction_init : fraction of nbterms to initialize large batch size : a good default value is between 0.01 and
    ///             0.02 large batch size begins at 0.01 * nbterms or 0.02 * nbterms.
    ///
    /// (see examples)
    ///
    pub fn new(
        eta_zero: f64,
        m_zero: f64,
        mini_batch_size_init: usize,
        large_batch_fraction_init: f64,
    ) -> StochasticControlledGradientDescent {
        //
        if large_batch_fraction_init > 1. {
            warn!("large_batch_size_init > 1. , fraction factor for large_batch size initialization must be < 1. , exiting");
            std::process::exit(1);
        }
        if m_zero > large_batch_fraction_init {
            warn!("m_zero > large_batch_size_init fraction , base fraction for nb mini batch should be less than for large_batch_size");
            std::process::exit(1);
        }
        info!(" eta_zero {:2.4E} m_zero {:2.4} \n mini batch size {:?} , large_batch_fraction_init {:2.4E}", 
            eta_zero, m_zero, mini_batch_size_init, large_batch_fraction_init);
        //
        StochasticControlledGradientDescent {
            rng: Xoshiro256PlusPlus::seed_from_u64(4664397),
            eta_zero: eta_zero,
            m_zero: m_zero,
            mini_batch_size_init: mini_batch_size_init,
            large_batch_fraction_init: large_batch_fraction_init as f64,
        }
    }
    /// Seeds the random number generator using the supplied `seed`.
    /// This is useful to create re-producable results.
    pub fn seed(&mut self, seed: [u8; 32]) {
        self.rng = Xoshiro256PlusPlus::from_seed(seed);
    }

    // batch growing factor cannot be too large, so it must be adjusted accoding to nbterms value, and nb_max_iterations
    // we choose the batch growing factor alfa so that :
    //     B_0 * alfa^(2*nb_max_iterations) = 1.
    // (B_0 is the fraction of nbterms we take at beginning of iterations)
    fn estimate_batch_growing_factor(&self, nb_max_iterations: usize, nbterms: usize) -> f64 {
        let batch_growing_factor: f64;
        //
        if self.m_zero * (nbterms as f64) < 1. {
            warn!("m_zero fraction , should be greater than 1./ number of terms in sum");
            std::process::exit(1);
        }
        //
        let log_alfa = (-self.large_batch_fraction_init.ln()) / (2. * nb_max_iterations as f64);
        batch_growing_factor = log_alfa.exp();
        if batch_growing_factor <= 1. {
            println!("batch growing factor shoud be greater than 1. , possibly you can reduce number of iterations ");
        }
        debug!(
            " upper bound for batch_growing_factor  {:2.4E}",
            batch_growing_factor
        );
        //
        return batch_growing_factor;
    } // end of estimate_batch_growing_factor

    // returns BatchSizeInfo for current iteration
    fn get_batch_size_at_jstep(
        &self,
        batch_growing_factor: f64,
        nbterms: usize,
        j: usize,
    ) -> BatchSizeInfo {
        let alfa_j = batch_growing_factor.powi(j as i32);
        // max size of large batch is 100 or 0.1 * the number of terms
        let max_large_batch_size;
        if nbterms > 100 {
            max_large_batch_size = (nbterms as f64 / 10.).ceil() as usize;
        } else {
            max_large_batch_size = nbterms;
        }
        // ensure max_mini_batch_size is at least 1.
        let max_mini_batch_size = (nbterms as f64 / 100.).ceil() as usize;
        // B_j
        let large_batch_size =
            ((self.large_batch_fraction_init * (nbterms as f64) * alfa_j * alfa_j).ceil() as usize)
                .min(max_large_batch_size);
        // b_j  grow slowly as log(1. + )
        let mini_batch_size =
            ((self.mini_batch_size_init as f64 * alfa_j).floor() as usize).min(max_mini_batch_size);
        // m_j  computed to ensure mean number of mini batch < large_batch_size as mini_batch_size_init < large_batch_size_init is enfored
        let nb_mini_batch_parameter = self.m_zero * (nbterms as f64) * alfa_j.powf(1.5);
        // let step decrease really slowly
        let step_size = self.eta_zero / alfa_j.sqrt();
        //
        BatchSizeInfo {
            _step: j,
            large_batch: large_batch_size,
            mini_batch: mini_batch_size,
            nb_mini_batch_parameter: nb_mini_batch_parameter,
            step_size: step_size,
        }
    } // end of get_batch_size_at_jstep
    ///
    /// sample number of mini batch according to geometric law of parameter p = b_j/(m_j+b_j)
    /// with law : P(N=k) = (1-p) * p^k. (proba of nb trial before success mode)
    fn get_nb_small_mini_batches(&self, batch_size_info: &BatchSizeInfo) -> usize {
        let m_j = batch_size_info.nb_mini_batch_parameter as f64;
        let b_j = batch_size_info.mini_batch as f64;
        // we return mean of geometric. Sampling too much instable due to large variance of geometric distribution.
        let mut n_j = (m_j / b_j).ceil() as usize;
        n_j = n_j.min(batch_size_info.large_batch);
        debug!(
            " nb small mini batch {:?} m_j {:2.4E} b_j : {:2.4E} ",
            n_j, m_j, b_j
        );
        return n_j;
    }
} // end impl StochasticControlledGradientDescent

// used to sample mini batches, Really faster.
//  1. Faster methods for Random Sampling J.S Vitter Comm ACM 1984
// 2. Kim-Hung Li Reservoir Sampling Algorithms : Comm ACM Vol 20, 4 December 1994
//  3. https://en.wikipedia.org/wiki/Reservoir_sampling

fn sample_without_replacement_reservoir(
    size_asked: usize,
    in_terms: &[usize],
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<usize> {
    let mut out_terms = Vec::<usize>::with_capacity(size_asked.min(in_terms.len()));
    for i in 0..size_asked {
        out_terms.push(in_terms[i]);
    }
    let mut xsi: f64;
    xsi = rand_distr::Standard.sample(rng);
    let mut w: f64 = (xsi.ln() / (size_asked as f64)).exp();

    let mut s = size_asked - 1;
    while s < in_terms.len() {
        xsi = rand_distr::Standard.sample(rng);
        s = s + (xsi.ln() / (1. - w).ln()).floor() as usize + 1;
        if s < in_terms.len() {
            // update random index in out_terms
            xsi = rand_distr::Standard.sample(rng);
            let idx = (size_asked as f64 * xsi).floor() as usize;
            out_terms[idx] = in_terms[s];
            // update w
            xsi = rand_distr::Standard.sample(rng);
            w = w * (xsi.ln() / (size_asked as f64)).exp();
        }
    }
    out_terms
}

impl<D: Dimension, F: SummationC1<D>> Minimizer<D, F, usize>
    for StochasticControlledGradientDescent
{
    type Solution = Solution<D>;

    fn minimize(
        &self,
        function: &F,
        initial_position: &Array<f64, D>,
        max_iterations: Option<usize>,
    ) -> Solution<D> {
        //
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        //
        let mut position = initial_position.clone();
        let mut value = function.value(&position);
        let nb_max_iterations = max_iterations.unwrap();
        // direction propagation
        let mut direction: Array<f64, D> = position.clone();
        direction.fill(0.);

        if log_enabled!(Info) {
            info!("Starting with y = {:e} \n for x = {:?}", value, position);
        } else {
            info!("Starting with y = {:e}", value);
        }
        trace!("nb_max_iterations {:?}", nb_max_iterations);
        //
        let mut iteration: usize = 0;
        let mut rng = self.rng.clone();
        let nb_terms = function.terms();
        let mut monitoring = IterationRes::<D>::new(nb_max_iterations, SolMode::Last);
        let batch_growing_factor =
            self.estimate_batch_growing_factor(nb_max_iterations, function.terms());
        // temporary gradients passed by ref to avoid possibly large reallocation
        let mut large_batch_gradient: Array<f64, D> = position.clone();
        large_batch_gradient.fill(0.);
        //
        let mut mini_batch_gradient_current: Array<f64, D>;
        mini_batch_gradient_current = position.clone();
        mini_batch_gradient_current.fill(0.);
        //
        let mut mini_batch_gradient_origin: Array<f64, D>;
        mini_batch_gradient_origin = position.clone();
        mini_batch_gradient_origin.fill(0.);
        //
        let all_indexes = Vec::<usize>::from_iter::<std::ops::Range<usize>>(0..nb_terms);
        // now we work
        loop {
            // get iteration parameters
            let iter_params =
                self.get_batch_size_at_jstep(batch_growing_factor, nb_terms, iteration);
            let n_j = self.get_nb_small_mini_batches(&iter_params);
            let step_size = iter_params.step_size;
            // sample large batch of size Bj
            let large_batch_indexes = sample_without_replacement_reservoir(
                iter_params.large_batch,
                &all_indexes,
                &mut rng,
            );
            trace!("\n iter {:?} got large batch size {:?}, nb mini batch {:?}, mini batch size {:?}, step {:2.4E}", 
                    iteration, large_batch_indexes.len(), n_j, iter_params.mini_batch, iter_params.step_size);
            // compute gradient on large batch index set and store initial position
            function.mean_partial_gradient(
                &position,
                &large_batch_indexes,
                &mut large_batch_gradient,
            );
            let position_before_mini_batch = position.clone();
            // loop on small batch iterations
            for _k in 0..n_j {
                // sample mini batch terms
                let terms = sample_without_replacement_reservoir(
                    iter_params.mini_batch,
                    &all_indexes,
                    &mut rng,
                );
                //
                function.mean_partial_gradient(&position, &terms, &mut mini_batch_gradient_current);
                //
                function.mean_partial_gradient(
                    &position_before_mini_batch,
                    &terms,
                    &mut mini_batch_gradient_origin,
                );
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
                direction = &mini_batch_gradient_current - &mini_batch_gradient_origin
                    + &large_batch_gradient;
                // step into the direction of the negative gradient
                position = position - step_size * &direction;
            } // end mini batch loop
              // update position
            iteration += 1;
            // some monitoring
            value = function.value(&position);
            let gradnorm = norm_l2(&direction);
            monitoring.push(value, &position, gradnorm);
            if log_enabled!(Debug) {
                debug!(
                    "\n Iteration {:?} y = {:2.4E} , gradient norm : {:2.4}",
                    iteration, value, gradnorm
                );
            }
            // convergence control or max iterations control
            if iteration >= nb_max_iterations {
                break;
            }
        } // end global loop
          //
        log::info!("\n StochasticControlledGradientDescent::minimize ; sys time(ms) {:?} cpu time(ms) {:?}", 
        sys_now.elapsed().unwrap().as_millis(),
        cpu_start.elapsed().as_millis());
        info!(
            "Reached maximal number of iterations required {:?}, stopping optimization",
            nb_max_iterations
        );
        let rank = monitoring.check_monoticity();
        info!(" monotonous convergence from rank : {:?}", rank);
        return Solution::new(position, value);
    } // end minimize
} // end impl impl<F: Summation1> Minimizer<F>

//==================================================================================

#[cfg(test)]
mod tests {

    //    use std::iter::FromIterator;
    use super::*;

    #[test]
    fn test_reservoir_sampling() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(4664397);
        let nb_asked = 100;
        let in_terms = Vec::<usize>::from_iter::<std::ops::Range<usize>>(0..60000);
        let selected_terms = sample_without_replacement_reservoir(nb_asked, &in_terms, &mut rng);
        //
        assert_eq!(selected_terms.len(), nb_asked);
        // sort and print
    }
} // end of mod tests
