CHANGES


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
6. Wrappers around undelying thrust algorithms using range semantics: gather, scatter, sort, sort_by_key, reduce, transform.
7. Counting, random and constant ranges:

  * `hydra::random_gauss_range(...)`,
  * `hydra::random_exp_range(...)`,
  * `hydra::random_flat_range(...)`,
  * `hydra::range(first, last)` 
  * `hydra::crange(value)`

8. Collecting range: `hydra::collect`

# Bug fixes

1. Null pointer breaking build in CLING
2. `multiarray::insert(...)`
