
//! This file is inspired by the crate optimisation written by  Oliver Mader b52@reaktor42.de.  
//! I kept the traits Function, FunctionC1, Summation and SummationC1 which provides
//! the interface for users to define a minimisation problem.
//! 
//! 
//! In fact when minimising summation function we often seek to minimize the mean of the terms
//! which is the same but scales gradient and this makes the implementation of batched stochastic gradient
//! more natural as we always computes mean gradient over terms taken into account.
//! 
//! I kept the names of methods and reimplemented according to the following changes:
//! 
//! 1. In batched stochastic gradient we need to define mean gradient on a subset of indexes. 
//! 2. I use the crate ndarray which provides addition of vector and enables rayon for //
//! 3. I got rid of the iterator on indexes as indexes are always usize.
//! 4. The function minimize in Trait Minimizer takes a generic Argument for future extension
//! 5. I use rayon to compute value of summation for values and gradients with parallel iterators
//! 



use rayon::prelude::*;

use ndarray::{Array, Dimension};


/// Defines an objective function `f` that is subject to minimization.
/// trait must be sync to be able to be sent over threads
pub trait Function<D:Dimension> : Sync {
    /// Computes the objective function at a given `position` `x`, i.e., `f(x) = y`.
    fn value(&self, position: &ndarray::Array<f64,D>) -> f64;
}



/// Defines an objective function `f` that is able to compute the first derivative
/// `f'(x)`.
pub trait FunctionC1<D:Dimension> : Function<D> {
    /// Computes the gradient of the objective function at a given `position` `x`,
    /// i.e., `∀ᵢ ∂/∂xᵢ f(x) = ∇f(x)`.
    fn gradient(&self, position: &Array<f64,D>) -> Array<f64,D>;
}


/// Defines a summation of individual functions, i.e., f(x) = ∑ᵢ fᵢ(x).
pub trait Summation<D:Dimension>: Function<D> {
    /// Returns the number of individual functions that are terms of the summation.
    fn terms(&self) -> usize;

    /// Comptues the value of one individual function indentified by its index `term`,
    /// given the `position` `x`.
    fn term_value(&self, position: &Array<f64,D>, term: usize) -> f64;

    /// Computes the partial sum over a set of individual functions identified by `terms`.
    /// without dividing by anything!!
    fn partial_value(&self, position:&Array<f64,D> , terms: &[usize]) -> f64 {
      //  let mut value = 0.0;
        let f = |t : usize| -> f64 { self.term_value(position, t)};
        let value = terms.into_par_iter().map(|t| f(*t)).sum();
        // for term in terms {
        //     value += self.term_value(position, *term);
        // }
        value
    }
} // end trait Summation


/// always return the mean of the sum
impl<D : Dimension, S: Summation<D> > Function<D> for S {
    fn value(&self, position: &Array<f64,D>) -> f64 {
        let value = self.partial_value(position, &(0..self.terms()).into_iter().collect::<Vec<usize>>());
        // 
        value/(self.terms() as f64)
    }
}


/// Defines a summation of individual functions `fᵢ(x)`, assuming that each function has a first
/// derivative.
pub trait SummationC1<D:Dimension> : Summation<D> + FunctionC1<D> {
    /// The required method the user must furnish.
    /// Computes the gradient of one individual function identified by `term` at the given
    /// `position` `without a 1/n renormalization`
    /// gradient index and position indexes must corrspond.
    fn term_gradient(&self, position: &Array<f64, D>, term: &usize, gradient : &mut Array<f64, D>);

    // gradient is passed as arg to avoid reaalocation!
    /// Computes the sum of partial gradient over a set of `terms` at the given `position`.
    /// `Wwithout a 1/n renormalization`
    fn partial_gradient(&self, position: &Array<f64, D>, terms: &[usize], gradient : &mut Array<f64, D>) {
        assert!(terms.len() > 0);
        gradient.fill(0.);
        let mut term_gradient : Array<f64, D> = gradient.clone();
        // could Rayon // here if length of iterator i.e dimension dimension of data is very large.
        if terms.len() < 1000 {
            for term in terms.into_iter() {
              self.term_gradient(position, &term, &mut term_gradient);
              *gradient += &term_gradient;
             }
        }
        else {
            // we do not use directly rayon beccause we want to avoid too many allocations of term_gradient
            // so we split iterations in blocks
            let block_size = 500;
            // to get start of i block
            let nb_block = if terms.len() % block_size == 0 { 
                    terms.len() / block_size
                 }
                 else {
                    1 + (terms.len() / block_size) 
            };
            let first = | i : usize | -> usize {
                let start = i * block_size;
                start
            };
            let last =  | i : usize | -> usize {
                let end = ((i+1) * block_size).min(terms.len());
                end
            };
            let compute = | i: usize | ->  Array<f64, D> {
                let mut block_gradient : Array<f64, D> = gradient.clone();
                block_gradient.fill(0.);
                let mut term_gradient : Array<f64, D> = gradient.clone();
                for k in first(i)..last(i) {
                    self.term_gradient(position, &terms[k], &mut term_gradient);
                    block_gradient += &term_gradient;
                }
                block_gradient
            }; // end compute function
            //
            // now we execute in // compute on each block
            //
          *gradient = (0..nb_block).into_par_iter().map(|b| compute(b)).reduce(|| gradient.clone(), | acc , g|  acc + g  );
        }
     } // end partial_gradient


    /// in batched stochastic gradient we need means of gradient on batch or minibatch
    fn mean_partial_gradient(&self, position: &Array<f64, D>, terms: &[usize], gradient : &mut Array<f64, D>)  {
        self.partial_gradient(position, terms, gradient);
        gradient.iter_mut().for_each(|x| *x /= terms.len() as f64);
    }  // end of mean_partial_gradient


}    // end trait SummationC1





impl<D:Dimension, S: SummationC1<D> > FunctionC1<D> for S {
    fn gradient(&self, position: &Array<f64, D>) -> Array<f64, D> {
        let mut gradient : Array<f64, D> = position.clone();
        gradient.fill(0.);
        let mut gradient_term : Array<f64, D> = position.clone();
        gradient_term.fill(0.);
        // CAVEAT : user parametrization ?
        if self.terms() < 2000 {
            for term in 0..self.terms() {
               self.term_gradient(position, &term, &mut gradient_term);
               gradient += &gradient_term;
             }
        }
        else {
            // we do not use directly rayon beccause we want to avoid too many allocations of term_gradient
            // so we split iterations in blocks
            let block_size = 1000;
            // to get start of i block
            let nb_block = if self.terms() % block_size == 0 { 
                    self.terms() / block_size
                 }
                 else {
                    1 + (self.terms() / block_size) 
            };
            let first = | i : usize | -> usize {
                let start = i * block_size;
                start
            };
            let last =  | i : usize | -> usize {
                let end = ((i+1) * block_size).min(self.terms());
                end
            };
            let compute = | i: usize | ->  Array<f64, D> {
                let mut block_gradient : Array<f64, D> = gradient.clone();
                block_gradient.fill(0.);
                let mut term_gradient : Array<f64, D> = gradient.clone();
                for k in first(i)..last(i) {
                    self.term_gradient(position, &k, &mut term_gradient);
                    block_gradient += &term_gradient;
                }
                block_gradient
            }; // end compute function
            //
            // now we execute in // compute on each block
            //
            gradient = (0..nb_block).into_par_iter().map(|b| compute(b)).reduce(|| gradient.clone(), | acc , g|  acc + g  );
        }
        //
        gradient.iter_mut().for_each(|x| *x /= self.terms() as f64);
        //
        gradient
    }

    
}  // end of impl<S: SummationC1> FunctionC1 for S


/// Defines an optimizer that is able to minimize a given objective function `F`.
pub trait Minimizer<D: Dimension, F: ?Sized, MinimizerArg> {
    /// Type of the solution the `Minimizer` returns.
    type Solution: Evaluation<D>;

    /// Performs the actual minimization and returns a solution.
    /// MinimizerArg should provide a number of iterations, a min error , or anything needed for implemented algorithm
    fn minimize(&self, function: &F, initial_position: &Array<f64,D>, args : Option<MinimizerArg>) -> Self::Solution;
}


/// Captures the essence of a function evaluation.
pub trait Evaluation<D:Dimension> {
    /// Position `x` with the lowest corresponding value `f(x)`.
    fn position(&self) -> &Array<f64,D>;

    /// The actual value `f(x)`.
    fn value(&self) -> f64;
}


/// A solution of a minimization run providing only the minimal information.
///
/// Each `Minimizer` might yield different types of solution structs which provide more
/// information.
#[derive(Debug, Clone)]
pub struct Solution<D:Dimension> {
    /// Position `x` of the lowest corresponding value `f(x)` that has been found.
    pub position: Array<f64,D>,
    /// The actual value `f(x)`.
    pub value: f64
}

impl <D:Dimension> Solution<D> {
    /// Creates a new `Solution` given the `position` as well as the corresponding `value`.
    pub fn new(position: Array<f64,D>, value: f64) -> Solution<D> {
        Solution {
            position: position,
            value: value
        }
    }
}

impl <D:Dimension> Evaluation<D> for Solution<D> {
    fn position(&self) -> &Array<f64,D> {
        &self.position
    }

    fn value(&self) -> f64 {
        self.value
    }
}





#[allow(dead_code)]
pub fn norm_l2<D:Dimension>(gradient : &Array<f64,D>) -> f64 {
    let norm = gradient.fold(0., |norm, x |  norm+ (x * x));
    norm.sqrt()
}
