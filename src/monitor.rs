//! This module stores result of iterations and can do some checks on convergence
//! possibly compute averaging of solutions along iterations
//!

use ndarray::{Array, Dimension};




/// an enum to represent computation mode of final solution
#[derive(Debug, Clone, PartialEq)]
pub enum SolMode {
    /// store only last position
    Last,
    /// store all positions
    Average,
}



/// a small utility to monitor convergence
/// This structure stores iteration at each step and possibly l2 norm of gradient
/// In Average mode each iteration step must store the position!
pub struct IterationRes<D:Dimension> {
    /// value of objective function at each step
    pub v_step : Vec<f64>, 
    /// l2 norm of gradient
    pub gradnorm : Vec<f64>,
        /// in average mode we must store each position (but it has a cost , beccause of clones)
    pub p_step : Option<Vec<Array<f64,D>>>,
    ///
    pub mode : SolMode,
    /// This stores the last position. It enables this structure to implement Solution trait
    /// even if positions where not all stored 
    pub last_position : Option<Array<f64,D>>,
}

// possibly return an average of iterates.

impl <D:Dimension> IterationRes<D> {

    /// initializes by number of iterations to store. 
    ///  - with_grad : bool to flag if we want to store gradients norm
    ///  Default initialization is solution is retrived from last value.
    pub fn new(max_nb_iter : usize, mode : SolMode) -> Self {
        let p_step = if mode != SolMode::Last { Some(Vec::<Array<f64,D>>::with_capacity(max_nb_iter))}
                     else { None
                     };
        //
        IterationRes {
            v_step : Vec::<f64>::with_capacity(max_nb_iter),
            gradnorm : Vec::with_capacity(max_nb_iter),
            p_step : p_step,
            mode : mode,
            last_position : None,
        }
    } // end of new

    ///
    pub fn push(&mut self, value:f64, position : &Array<f64,D>, gradient : f64) {
        //
        self.v_step.push(value);
        if self.mode == SolMode::Average {
            self.p_step.as_mut().unwrap().push(position.clone());
            panic!("in average mode position must not be none")
        }
        self.gradnorm.push(gradient);
    }

} // end of block impl


 
impl <D:Dimension> IterationRes<D> {
    /// returns rank of iter from which value is monotone decreasing.
    pub fn check_monoticity(&self) -> usize {
        let mut last = self.v_step.len() -1;
        while last >= 1 {
            if self.v_step[last-1] < self.v_step[last] {
                break;
            }
            last = last -1;
        };
        return last+1;
    }
}





