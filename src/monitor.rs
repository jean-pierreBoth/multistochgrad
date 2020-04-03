//!
//!



/// a small utility to monitor convergence
/// 
pub struct IterRes {
    /// value of objective function
    pub value : f64,
    /// l2 norm of gradient
    pub gradnorm : f64,
}
