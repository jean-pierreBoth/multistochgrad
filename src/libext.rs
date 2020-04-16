//! A module to define exterior C binding  language Julia in fact, interface
//

use std::os::raw::*;


extern crate env_logger;
extern crate rand;
extern crate rand_distr;



use ndarray::prelude::*;


use crate::types::*;
use crate::scsg::*;

/// For Julia the API will need to pass functions defining
/// trait Function and Summation. 
// recall the following from std::fn doc : In addition, function pointers of any signature, ABI, or safety are Copy, 
// and all safe function pointers implement Fn, FnMut, and FnOnce.


/// This is the Objective functio we minimize
/// It takes a pointer to a position , length of position vector   then return a score
/// 
type ValueFnPtr = extern "C" fn(*const f64, len : c_ulong) -> f64;


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


//=========================================
/// 


#[repr(C)]
/// structure exported to foreign language and passing argument to StochasticControlledGradientDescent
pub struct SCSG_Ffi {
    /// step
    pub(crate) eta_zero : f64,
    /// fraction of nbterms to consider in initialization of mⱼ governing evolution of nb small iterations
    pub(crate) m_zero: f64,
    /// m_0 in the paper
    pub(crate) mini_batch_size_init : u64,
    /// related to B_0 in Paper. Fraction of terms to consider in initialisation of B_0
    pub(crate) large_batch_size_init: f64,
    ///
    pub(crate) nb_iter: u64,
}




//===========================================================================================

#[repr(C)]
pub struct SVRG_Ffi {

}





#[repr(C)]
pub struct SAG_Ffi {

}


#[repr(C)]
/// The structure returned to C/Julia
pub struct FfiSolution {
    /// solution
    value : f64,
    /// position solution
    position : *const f64,
    /// len of position (in fact known by client)
    len : u64,
}

impl FfiSolution {
    fn new(value:f64, position : &Array1<f64>) -> Self {
        let len = position.len();
        // here , if we were to use Array2 we would have to consider that Rust ndarray has row order
        // as C , but Julia has column order as Fortran!!!
        let ptr = position.as_ptr();
        std::mem::forget(position);
        FfiSolution {
            value : value,
            position : ptr,
            len : len as u64,
        }
    } // end of new

}  // end impl FfiSolution



pub extern "C" fn minimize_scsg(scsg_pb : *const SCSG_Ffi, to_minimize : &FfiProblem, pos_ptr : *const f64, len_pos : u64) -> *const FfiSolution {
    // allocate StochasticControlledGradientDescent
    let eta_zero;
    let m_zero;
    let mini_batch_size_init;
    let large_batch_size_init;
    let nb_iter;
    let initial_position: Array1<f64>;
    //
    unsafe {
        eta_zero = (*scsg_pb).eta_zero;
        m_zero = (*scsg_pb).m_zero;
        mini_batch_size_init = (*scsg_pb).mini_batch_size_init as usize;
        large_batch_size_init = (*scsg_pb).large_batch_size_init;
        nb_iter = (*scsg_pb).nb_iter as usize;
        // reconstruct initial_position
        let slice = std::slice::from_raw_parts(pos_ptr, len_pos as usize);
        let data_v = Vec::from(slice);
        initial_position = Array1::<f64>::from(data_v);
    }
    let scsg_pb  = StochasticControlledGradientDescent::new(eta_zero, m_zero, mini_batch_size_init,
                                                         large_batch_size_init);
    // solve minimization pb
    let solution = scsg_pb.minimize(to_minimize, &initial_position , Some(nb_iter));
    // convert boack to FfiSolution
    let ffi_solution = FfiSolution::new(solution.value(), solution.position());
    //
    return Box::into_raw(Box::new(ffi_solution));        
}