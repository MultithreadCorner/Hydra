## CHANGE LOG

### Hydra 2.3.1 

1. Interfaces to FFTW and CuFFT for performing 1D real-real complex-real and real-complex FFT on CPU and GPU 
2. FFT based convolution on CPU and GPU for arbitrary pair of functors: `hydra::convolute` and `hydra::ConvolutionFunctor`
3. Cubic spiline reimplementation for deal with abitrary large datasets: `hydra::spiline` and `hydra::SpilineFunctor`
4. new examples showing how to deploy convolution in fits, spilines and FFT.
5. Many bug fixes across the tree...


### Hydra 2.2.1 (probably incomplete)

1. New functors and implementations (`hydra/functions`):

    * ArgusShape
    * BifurcatedGaussian
    * BlattWeisskopfFunctions
    * BreitWignerLineShape
    * BreitWignerNR
    * Chebychev
    * ChiSquare
    * CosHelicityAngle
    * CrystalBallShape
    * DeltaDMassBackground
    * Exponential
    * Gaussian
    * GaussianKDE
    * GeneralizedGamma
    * Ipatia
    * JohnsonSUShape
    * LogNormal
    * M12PhaseSpaceLineShape
    * M12SqPhaseSpaceLineShape
    * PlanesDeltaAngle
    * Polynomial
    * ThreeBodyMassThresholdBackground
    * TrapezoidalShape
    * TriangularShape
    * UniformShape
    * WignerDMatrix
    * ZemachFunctions


2. Orthogonal polynomials (`hydra/functions/Math.h`)

    * Chebychev of 1st and 2nd kinds 
    * Laguerre
    * Hermite
    * Legendre
    * Jacobi

3. New `Parameter::Create("name")` method overload.
4. Wrappers around thrust algorithms using range semantics: 

    * gather
    * scatter
    * sort
    * sort_by_key 
    * reduce
    * transform. 
    * See header `hydra/Algorithms.h`

5. Predefined ranges:

    *  `hydra::random_gauss_range(...)`,
    *  `hydra::random_exp_range(...)`,
    *  `hydra::random_flat_range(...)`,
    *  `hydra::range(...)` 
    *  `hydra::constant_range(...)`
    *  `hydra::phase_space_range(...)`

6. Collecting range: `hydra::collect` to reorder a range of values according to a indexing scheme.
7. Introduction of `hydra::make_loglikelihood_fcn` overloads supporting range semantics. 
8. Introduction of `hydra::make_loglikelihood_fcn` overloads for binned datasets.
9. Implementation of `hydra::begin`, `hydra::end`, `hydra::begin`, `hydra::rend` free functions.
10. Range semantics for hydra::PhaseSpace.

# Bug fixes

1. Null pointer breaking build in CLING
2. Fix syntax error in `multiarray::insert(...)`
3. Fix syntax error in `multivector::insert(...)`
4. Fix syntax error in `hydra::reduce`
5. Fix syntax error in `hydra::make_loglikelihood_fcn` overloads supporting weighted datasets.
