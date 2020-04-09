# multistochgrad

This crate provides a Rust implementation of some stochastic gradient algorithms.

The algorithms implemented here are dedicated to the minimization of objective function represented by the
mean of many functions as occurring in learning contexts.

The implemented algorithms are:

* The so-called SCSG algorithm described and analyzed in the two papers by L. Lei and  M.I Jordan.

    1. "On the adaptativity of stochastic gradient based optimization" (2019)
    [SCSG-1](https://arxiv.org/abs/1904.04480)

    2. "Less than a single pass : stochastically controlled stochastic gradient" (2019)
    [SCSD-2](https://arxiv.org/abs/1609.03261)

* The SVRG algorithm described in the paper by R. Johnson and T. Zhang
"Accelerating Stochastic Gradient Descent using Predictive Variance Reduction" (2019).  
[Advances in Neural Information Processing Systems, pages 315–323, 2013](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf)

* The SAG algorithm described in : 
 
The Stochstic Averaged Gradient Descent as described in the paper:
"Minimizing Finite Sums with the Stochastic Average Gradient" (2013, 2016)
M.Schmidt, N.LeRoux, F.Bach

All algorithms alternates some form of large batch computation(computing gradient of many terms of the sum)
and small or mini batches (computing a small number of terms, possibly just one, term of the gradient)
and updating position by combining these global and local gradients.

Further details on the algorithms are in the doc of files scsg.rs, svrg.rs, sag.rs
and in the reference papers.

The implementation is based on the ndarray and rayon crates that provide respectiveley efficient
array manipulation and transparent threaded computations.

## Examples and tests

Examples are based on logisitc regression applied to digits MNIST database
(as in the second paper on SCSG). The database has 60000 images of handwritten digits of 784 pixels.  
The logistic regression is tested with the 3 algorithms and some comments are provided, comparing the results.

Small tests consist in a line fitting problem that is taken  from the crate optimisation.

## Acknowledgement

This crate is indebted to the crate optimisation from which I kept the traits `Function`, `Summation`
defining the user interface after various modifications which are detailed in the file ``types.rs``

## License

Licensed under either of

* Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
* MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.

This software was written on my own while working at [CEA](http://www.cea.fr/), [CEA-LIST](http://www-list.cea.fr/en/)

## Building

By default the crate is a standalone project and builds a static libray and executable.
To be used with the companion Julia package it is necessary to build a dynamic library.
This can be done by just uncommenting (i.e get rid of the #) in file Cargo.toml the line:

*#crate-type = ["dylib"]*

and rerun the command: cargo build --release.

This will generate a .so file in the target/release directory.
