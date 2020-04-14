//! A module to define exterior C binding  language Julia in fact, interface
//

use std::os::raw::*;


extern crate env_logger;
extern crate rand;
extern crate rand_distr;



use ndarray::prelude::*;


use crate::types::*;

/// For Julia the API will need to pass functions defining
/// trait Function and Summation




/// This type is for function with a C-API
/// It takes a pointer to a position , length of position vector, and a term number,  thenreturn a score
/// 
type TermValueFnPtr = extern "C" fn(*const f64, len : c_ulong, term : c_ulong) -> f64;



/// This type corresponds to term radient computation.
/// first pointer is position, second pointer corresponds to a preallocated pointer for return value.
/// len is size of both (equal) allocated arrays,
/// term is term for which we compute gradient
type TermGradientFnPtr = extern "C" fn(pos : *const f64, grad : *const f64, len : c_ulong, term : c_ulong) -> c_ulong;


/// The structure describing a minimization problem form a FFI interface
/// We do not have observations, we only have pointer to functions
/// 
pub struct FfiProblem {
    /// nbterms of the problem
    nbterms : usize,
    /// computes value of a term
    term_value_f : TermValueFnPtr,
    /// computes  gradient value  of a term
    term_gradient_f : TermGradientFnPtr,
}


impl FfiProblem {
    pub fn terms(&self) -> usize { self.nbterms }

} // end of impl block for FfiProblem
 
impl Summation<Ix1> for FfiProblem {
    /// terms implementation
    fn terms(&self) -> usize {
        self.terms()
    }
    ///compute term component in sum
    fn term_value(&self, w: &Array1<f64>, i: usize) -> f64 {
        // get back to function pointer
        let v = (self.term_value_f)(w.as_ptr(),w.len() as u64, i as u64);
        //
        v
    }
}
// partial value of function




impl SummationC1<Ix1> for  FfiProblem {
    /// computes partial gradient
    fn term_gradient(&self, w: &Array1<f64>, term: &usize, gradient : &mut Array1<f64>)  {
        let len = w.len();
        let pos_ptr = w.as_ptr();
        let grad_ptr = gradient.as_mut_ptr();
        (self.term_gradient_f)(pos_ptr, grad_ptr, len as u64, *term as u64);
    }
}
