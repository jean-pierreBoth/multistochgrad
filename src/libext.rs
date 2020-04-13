//! A module to define exterior language Julia in fact, interface
//

use std::os::raw::*;


/// For Julia the API will need to pass functions defining
/// trait Function and Summation


/// This type is for function with a C-API
/// It takes a pointer to a position , length of position vector and return a score
/// 
type ValueFnPtr = extern "C" fn(*const f64, len : c_ulong) -> f64;


/// This type corresponds to Gradient computation.
/// first pointer is position, second pointer corresponds to preallocated pointer for return value.
/// len is size of both (equal) allocated arrays.
type GradientFnPtr = extern "C" fn(*const f64, *const f64, len : c_ulong) -> f64;



// partial value of function


// partial gradient