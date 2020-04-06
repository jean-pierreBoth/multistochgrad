# multistochgrad

This crate provides a Rust implementation of some stochastic gradient algorithms.

* The so-called SCSG algorithm described and analyzed in the two papers by by L. Lei and  M.I Jordan.

    1. "On the adaptativity of stochastic gradient based optimization" (2019)
    [SCSG-1](https://arxiv.org/abs/1904.04480)

    2. "Less than a single pass : stochastically controlled stochastic gradient" (2019)
    [SCSD-2](https://arxiv.org/abs/1609.03261)

* The SVRG algorithm described in the paper by R. Johnson and T. Zhang
"Accelerating Stochastic Gradient Descent using Predictive Variance Reduction" (2019). 
[Advances in Neural Information Processing Systems, pages 315–323, 2013](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf)

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
