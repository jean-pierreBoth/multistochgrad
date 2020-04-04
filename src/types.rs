// Copyright (c) 2016 Oliver Mader <b52@reaktor42.de>
//
//! This file is inspired by the crate optimisation written by b52@reaktor42.de
// We kept the traits Function, FunctionC1, Summation and SummationC1
// and changed slightly the function signatures.
// 1. We use the crate ndarray with its dependancy rayon for //
// 2. In batched stochastic gradient we need to define mean gradient on a subset
//    of indexes. 
// 3. We get rid of the iterator on indexes as indees are always usize.
//! In fact we minimize the mean of the summation which is the same but scales gradient.


use rayon::prelude::*;

use ndarray::{Array, Dimension};


// Defines an objective function `f` that is subject to minimization.
//
pub trait Function<D:Dimension> : Sync {
    /// Computes the objective function at a given `position` `x`, i.e., `f(x) = y`.
    fn value(&self, position: &ndarray::Array<f64,D>) -> f64;
}


/// New-type to support optimization of real functions without requiring
/// to implement a trait.
// pub struct Func<D,F> 
//        where  F : Fn(&Array<f64,D>) -> f64,
//        D : Dimension
// {
//     pub f : F,
// }

// impl<D : Dimension, F: Fn(&Array<f64,D>) -> f64> Function<D> for Func<D, F> {
//     fn value(&self, position: &Array<f64,D>) -> f64 {
//         (self.f)(position)
//     }
// }


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


impl<D : Dimension, S: Summation<D> > Function<D> for S {
    fn value(&self, position: &Array<f64,D>) -> f64 {
        self.partial_value(position, &(0..self.terms()).into_iter().collect::<Vec<usize>>())
    }
}


/// Defines a summation of individual functions `fᵢ(x)`, assuming that each function has a first
/// derivative.
pub trait SummationC1<D:Dimension> : Summation<D> + FunctionC1<D> {
    /// The required method the user must furnish.
    /// Computes the gradient of one individual function identified by `term` at the given
    /// `position`. gradient index and position indexes must corrspond.
    fn term_gradient(&self, position: &Array<f64, D>, term: &usize, gradient : &mut Array<f64, D>);

    // gradient is passed as arg to avoid reaalocation!
    /// Computes the partial gradient over a set of `terms` at the given `position`.
    fn partial_gradient(&self, position: &Array<f64, D>, terms: &[usize], gradient : &mut Array<f64, D>) {
        assert!(terms.len() > 0);
        gradient.fill(0.);
        let mut term_gradient : Array<f64, D> = gradient.clone();
        // could Rayon // here if length of iterator i.e dimension dimension of data is very large.
        for term in terms.into_iter() {
            self.term_gradient(position, &term, &mut term_gradient);
            *gradient += &term_gradient;
        }
     } // end partial_gradient


    /// in batched stochastic gradient we need means of gradient on batch or minibatch
    fn mean_partial_gradient(&self, position: &Array<f64, D>, terms: &[usize], gradient : &mut Array<f64, D>)  {
        gradient.fill(0.);
        self.partial_gradient(position, terms, gradient);
        gradient.iter_mut().for_each(|x| *x /= self.terms() as f64);
    }  // end of mean_partial_gradient


}    // end trait SummationC1





impl<D:Dimension, S: SummationC1<D> > FunctionC1<D> for S {
    fn gradient(&self, position: &Array<f64, D>) -> Array<f64, D> {
        let mut gradient : Array<f64, D> = position.clone();
        gradient.fill(0.);
        let mut gradient_term : Array<f64, D> = position.clone();
        gradient_term.fill(0.);
        // CAVEAT to //
        for term in 0..self.terms() {
            self.term_gradient(position, &term, &mut gradient_term);
            gradient += &gradient_term;
        }
        gradient.iter_mut().for_each(|x| *x /= self.terms() as f64);
        //
        gradient
    }

    
}  // end of impl<S: SummationC1> FunctionC1 for S


/// Defines an optimizer that is able to minimize a given objective function `F`.
pub trait Minimizer<D: Dimension, F: ?Sized> {
    /// Type of the solution the `Minimizer` returns.
    type Solution: Evaluation<D>;

    /// Performs the actual minimization and returns a solution that
    /// might be better than the initially provided one.
    fn minimize(&self, function: &F, initial_position: &Array<f64,D>, nbiter:usize) -> Self::Solution;
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
