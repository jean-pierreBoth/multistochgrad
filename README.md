# Multistochgrad

This crate provides a Rust implementation of some stochastic gradient algorithms.

The algorithms implemented here are dedicated to the minimization of (convex) objective function represented by the
mean of many functions as occurring in various statistical and learning contexts.

The implemented algorithms are:

1. The so-called SCSG algorithm described and analyzed in the two papers by L. Lei and  M.I Jordan.

    * "On the adaptativity of stochastic gradient based optimization" (2019)
    [SCSG-1](https://arxiv.org/abs/1904.04480)

    * "Less than a single pass : stochastically controlled stochastic gradient" (2019)
    [SCSG-2](https://arxiv.org/abs/1609.03261)

2. The SVRG algorithm described in the paper by R. Johnson and T. Zhang
"Accelerating Stochastic Gradient Descent using Predictive Variance Reduction" (2013).  
[Advances in Neural Information Processing Systems, pages 315–323, 2013](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf)

3. The Stochastic Averaged Gradient Descent (SAG) as described in the paper:
"Minimizing Finite Sums with the Stochastic Average Gradient" (2013, 2016)
M.Schmidt, N.LeRoux, F.Bach

These algorithms minimize functions given by an expression:

        f(x) = 1/n ∑ fᵢ(x) where fᵢ is a convex function.

The algorithms alternate some form of large batch computation (computing gradient of many terms of the sum)
and small or mini batches (computing a small number of terms, possibly just one, term of the gradient)
and updating position by combining these global and local gradients.




## Examples and tests

Small tests consist in a line fitting problem that is taken  from the crate optimisation.

Examples are based on logisitc regression applied to digits MNIST database
(as in the second paper on SCSG).  
The data files can be downloaded from [MNIST](http://yann.lecun.com/exdb/mnist).

The logistic regression, with 10 classes,  is tested with the 3 algorithms and some comments are provided, comparing the results.

Run times are obtained on a i9-13900HX (32 threads). We give wall clock time and cpu times spent in minimizer.


### SCSG logistic regression

For the signification of the parameters B_0 , m_O, see documentation of SCSG. b_0 was set to 1
in all the runs.

Here we give some results:

* initialization position : 9 images with *constant pixel = 0.5*,
error at initial position: 6.94

| nb iter |  B_0  |  m_0  | step_0 | error | time(s) | cpu time(s) |
| :-----: | :---: | :---: | :----: | ----- | ------- | ----------- |
|   50    | 0.015 | 0.004 |  0.1   | 0.285 | 2.9     | 14.8        |
|   50    | 0.015 | 0.006 |  0.1   | 0.279 | 6.8     | 19          |
|   100   | 0.02  | 0.004 |  0.1   | 0.266 | 7.89    | 32.5        |
|   50    | 0.02  | 0.004 |  0.1   | 0.289 | 3.89    | 16          |
|   150   | 0.02  | 0.004 |  0.1   | 0.257 | 11      | 50          |


* initialization position : 9 images with *constant pixel = 0.0*,
error at initial position: 2.3

| nb iter | B_0   | m_0   | step_0 | error | time(s) | cpu time(s) |
| ------- | ----- | ----- | ------ | ----- | ------- | ----------- |
| 50      | 0.015 | 0.004 | 0.1    | 0.274 | 4.7     | 17          |
| 50      | 0.02  | 0.004 | 0.1    | 0.277 | 3.7     | 16.5        |
| 50      | 0.02  | 0.006 | 0.1    | 0.267 | 5.5     | 18          |
| 100     | 0.02  | 0.004 | 0.1    | 0.260 | 7.6     | 33          |


Increasing parameter controlling the number of minibatch decrease parallelism.  
It seems that convergence from the initialization from a null image is slightly easier than
with a constant 0.5 pixel.

### SVRG logistic regression

* initialization position : 9 images with *constant pixel = 0.5*,
error at initial position: 6.94

| nb iter | nb mini batch | step | error | time(s) | cpu time(s) |
| ------- | :-----------: | ---- | :---: | :-----: | ----------- |
| 100     |     1000      | 0.02 | 0.269 |  10.5   | 159         |
| 25      |     1000      | 0.05 | 0.288 |   2.6   | 40          |
| 50      |     1000      | 0.05 | 0.263 |   5.    | 81          |
| 100     |     1000      | 0.05 | 0.249 |  10.2   | 160         |

* initialization position : 9 images with *constant pixel = 0.0*,
error at initial position: 2.3

| nb iter | nb mini batch | step | error | time(s) | cpu time(s) |
| ------- | :-----------: | ---- | ----- | ------- | ----------- |
| 50      |     1000      | 0.05 | 0.258 | 5.3     | 80          |
| 50      |     2000      | 0.05 | 0.247 | 7.5     | 81          |
| 100     |     1000      | 0.05 | 0.247 | 10      | 161         |



### SAG logisitc regression

* initialization position : 9 images with *constant pixel = 0.5*,
error at initial position: 6.94

| nb iter | batch size | step  | error | time(s) | cpu time(s) |
| :-----: | :--------: | :---: | :---: | ------- | ----------- |
|  1000   |    1000    |  0.2  | 0.47  | 17      | 272         |
|  1000   |    1000    |  0.5  | 0.35  | 17      | 273         |
|  1000   |    2000    |  0.5  | 0.34  | 17.6    | 262         |
|  2000   |    1000    |  0.5  | 0.297 | 34.6    | 546         |

### Results

Tests show that the SCSG  outperforms SVRG by a factor 1.5 or 2  at equivalent precision in
both case with a correct initialization and one far from the solution.
SVRG clearly outperforms SAG.  
SCSG is very fast at reaching a good approximation roughly 0.28 even though it never runs on
the whole (one tenth) in this implementation. 
SCSG needs larger problem to benefit from multithreading.  

## Acknowledgement

This crate is indebted to the crate **optimisation** from which I kept the traits `Function`, `Summation`
defining the user interface after various modifications which are detailed in the file ``types.rs``


## License

Licensed under either of

* Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
* MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.

