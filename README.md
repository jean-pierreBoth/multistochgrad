# Multistochgrad

**WARNING : This is a preliminary version of a crate not yet "published" in Rust crate repository**.

This crate provides a Rust implementation of some stochastic gradient algorithms.

The algorithms implemented here are dedicated to the minimization of objective function represented by the
mean of many functions as occurring in various statistical and learning contexts.

The implemented algorithms are:

* The so-called SCSG algorithm described and analyzed in the two papers by L. Lei and  M.I Jordan.

    1. "On the adaptativity of stochastic gradient based optimization" (2019)
    [SCSG-1](https://arxiv.org/abs/1904.04480)

    2. "Less than a single pass : stochastically controlled stochastic gradient" (2019)
    [SCSD-2](https://arxiv.org/abs/1609.03261)

* The SVRG algorithm described in the paper by R. Johnson and T. Zhang
"Accelerating Stochastic Gradient Descent using Predictive Variance Reduction" (2013).  
[Advances in Neural Information Processing Systems, pages 315–323, 2013](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf)

* The SAG algorithm described in :

The Stochastic Averaged Gradient Descent as described in the paper:
"Minimizing Finite Sums with the Stochastic Average Gradient" (2013, 2016)
M.Schmidt, N.LeRoux, F.Bach

All algorithms alternates some form of large batch computation (computing gradient of many terms of the sum)
and small or mini batches (computing a small number of terms, possibly just one, term of the gradient)
and updating position by combining these global and local gradients.

Further details on the algorithms are in the doc of files scsg.rs, svrg.rs, sag.rs
(run **cargo doc --no-deps** as usual to get html generated docs) and in the reference papers.

The implementation is based on the `ndarray` and `rayon` crates that provide respectiveley efficient
array manipulation and transparent threaded computations. All batch computation of size greater than 1000 terms
are multithreaded.

## Examples and tests

Small tests consist in a line fitting problem that is taken  from the crate optimisation.

Examples are based on logisitc regression applied to digits MNIST database
(as in the second paper on SCSG). The database has 60000 images of 784 pixels corresponding to
handwritten digits form 0 to 9.  
The logistic regression, with 10 classes,  is tested with the 3 algorithms and some comments are provided, comparing the results.
Times are obtained by launching twice the example to avoid the compilation time of the first pass.
Run times are those obtained on a 4 i7 hyperthreaded cores laptop at 2.7Ghz

### SCSG logistic regression

For the signification of the parameters B_0 , b_O, see documentation of SCSG.
Here we give some results:

* initialization position : 9 images with *constant pixel = 0.5*,
error at initial position: 6.94

| nb iter | B_0    |   b_0    | step    | y value | time(s) |
|  :---:  |:---:  |  :-----:  | :----:  |   ----  |  ----   |
| 100     | 0.015  |  0.0015  |  0.5    |  0.49   |  13.4  |
| 100     | 0.010  |  0.0015  |  0.5    |  0.50   |  14.0  |
| 200     | 0.015  |  0.0015  |  0.5    |  0.37   |  26    |
| 100     | 0.015  |  0.0015  |  0.25   |  0.65   |  13.4  |
|  50     | 0.015  |  0.0015  |  0.5    |  0.75   |  7.0   |

* initialization position : 9 images with *constant pixel = 0.0*,
error at initial position: 2.3

| nb iter | B_0    |   b_0    | step   | y value  | time(s) |
|  ---    |----    |  ----    | ------ |   ----   |  ----  |
|  200    | 0.015  |  0.0015  |  0.5    |  0.315  |  26    |
|  100    | 0.015  |  0.0015  |  0.5    |  0.356  |  13    |
|  50     | 0.015  |  0.0015  |  0.5    |  0.42   |  6.8   |

### SVRG logistic regression

* initialization position : 9 images with *constant pixel = 0.0*,
error at initial position: 2.3

| nb iter |  nb mini batch     | step    | y value  | time(s) |
|  ---    |     :---:          | ------  |   ----   |  ----  |
|  50     |     500            |  0.0015 |  0.46    |  20    |
|  50     |     1000           |  0.0015 |  0.40    |  21    |
|  50     |     2000           |  0.0015 |  0.42    |  23    |
|  50     |     1000           |  0.005  |  0.52    |  21    |

We see between the first and the second line that running mini batch of one randomly selected one
comes for free in cpu time but the precision is better.
The third line shows that with too many mini batch we do not gain any more. In fact
monitoring the convergence shows that we begins to obsrve instability due to the length of mini batches
(in fact we had y = 0.41 at iteration 33)
Last line shows that the step must be small.

### SAG logisitc regression

* initialization position : 9 images with *constant pixel = 0.5*,
error at initial position: 6.94

| nb iter |  batch size  | step   | y value  | time(s) |
|  :---:  |  :---:       |  :---: | ------   |   ----  |
|  2000   |  1000        |  0.1   |  0.47    |   80    |
|  1000   |  2000        |  0.1   |  0.90    |   45    |
|  2000   |  2000        |  0.1   |  0.47    |   90    |
|  2000   |  2000        |  0.2   |  0.37    |   90    |
|  2000   |  1000        |  0.2   |  0.37    |   80    |

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
