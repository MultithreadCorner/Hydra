rImplementation ##CHANGES


### Hydra 2.2.0

# New features

1. ChiSquare functor and analytical integral
2. Orthogonal polynomials (`hydra/functions/Math.h`)

  * Chebychev 1st and 2nd kinds 
  * Laguerre
  * Hermite
  * Legendre
3. 1D kernel density estimation functor: GaussianKDE
4. Bifurcated Gaussian functor:  BifurcatedGaussian
5. `Parameter::Create("name")` method
6. Wrappers around thrust algorithms using range semantics: gather, scatter, sort, sort_by_key, reduce, transform. See header `hydra/Algorithms.h`
7. Counting, random and constant ranges:

  *  `hydra::random_gauss_range(...)`,
  *  `hydra::random_exp_range(...)`,
  *  `hydra::random_flat_range(...)`,
  *  `hydra::range(first, last)` 
  *  `hydra::crange(value)`

8. Collecting range: `hydra::collect` to reorder a range of values according to a indexing scheme.
9. Introduction of `hydra::make_loglikelihood_fcn` overloads supporting range semantics. 
10. Introduction of `hydra::make_loglikelihood_fcn` overloads for binned datasets.
11. Implementation of `hydra::begin`, `hydra::end`, `hydra::begin`, `hydra::rend`.
 

# Bug fixes

1. Null pointer breaking build in CLING
2. Fix syntax error in `multiarray::insert(...)`
3. Fix syntax error in `multivector::insert(...)`
