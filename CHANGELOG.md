## CHANGE LOG

### Hydra 3.2.2

This release:

1) 

Bug fixes:

1) Missing assignment operators for hydra::FunctionArgments

2) Covering composite of composites (https://github.com/MultithreadCorner/Hydra/issues/100)

3) Covering caching for parameterless functors.


### Hydra 3.2.1

This release:

1) In certain corner cases, a substantial part of the calculations performed to evaluate a functor depends only on the functor's state and it's parameters; i.e. it does not depend on the current functor's arguments that can be dataset points, numerical integration abscissas and so on. To further optimize the calculations in these cases, Hydra now provides the method ``virtual void hydra::Parameter::Update(void)``, which can be overridden by the functors 
in order to pre-calculate the factors only depending in the parameters, before the calculations are distributed to one of the parallel backends.  

2) Introduction of `hydra::random_range` and retirement of `hydra::random_uniform_range`, `hydra::random_gaussian_range`, `hydra::random_exp_range` , to define iterators for drawing samples from functors. Examples updated accordingly.


Bug fixes:

1) ROOT examples updated and tested against ROOT/6.22.00. ( https://github.com/MultithreadCorner/Hydra/commit/acd303e7fb21d220beb7986ab7122206ac6a8eed )

2) Correction of `hydra::CrystalBallShape` math. ( https://github.com/MultithreadCorner/Hydra/commit/a7ce56d4c9efad46351ddee630f7afce245c9909 )

3) Spelling of some words (in code) accross the tree. (https://github.com/MultithreadCorner/Hydra/commit/d2161898bd9ed4e09ee6593d27372c8ad43347b1) 

4) Fixing fallback path in `MSB.h`. ( https://github.com/MultithreadCorner/Hydra/commit/5b10e05e2517c27f270507c27ea76c85a62e24f5 )

5) Reimplementation of `hydra::detail::is_valid_type_pack` ( https://github.com/MultithreadCorner/Hydra/commit/f30b97a414f513e3e410195555bd0eb8f44eb9f9 )


-------------------------------------


### Hydra 3.1.0

This release substantially expands the set of pseudorandom number generators available in Hydra.
From Random123 (see: *John K. Salmon and others, (2011) "Parallel random numbers: as easy as 1, 2, 3"*. https://dl.acm.org/doi/10.1145/2063384.2063405), Hydra provides wrappers and implementations for 

1) *Philox* 
2) *Ars*
3) *Threefry*

*Squares* PRNG ( see: *Widynski, Bernard (2020). "Squares: A Fast Counter-Based RNG"*. https://arxiv.org/abs/2004.06278v2 ), are available from now, in two versions:

1) *Squares3*
2) *Squares4*

All the new generators belong to the *count-based family*, have excelent statistical properties, passing BigCrush (TestU01) and other tests easily, without any failure. All implementations provide very long periods (2^64 -1 or higher). For Squares{3,4}, users get a set of 2500 precalculated seeds for generation of sequences (streams) without statistical artifacts among them (all credits to Bernard Widynski!).

In summary, ***Squares3, Squares4 and Philox are way faster*** than the any option available in the previous releases. Ars and Threefry are competitive, being most of the time slightly faster than the fastest native `Thrust` rng. 

**From this release, the defaut PRNG in Hydra is set to hydra::squares3**. 

#### General

* Bug fixed in ```hydra::unweight``` implementation.
* Other minor fixes and modifitions across the tree.

______________________________


### Hydra 3.0.0

It is the first release of the longly  waited 3 series. Overall, this release is expected to be faster
or at least to have similar performance to the previous releases. 
There are many bug fixes and other changes across. The most significant are summarized below:

#### C++14 compiliant release

This release is the first C++14 compatible release. So, move the versions of NVCC, GCC, CLANG and so on, acordinaly.
Also, set the "--std" compiler flags to "--std=c++14" for both CUDA and host compilers. 
The first the minimal **CUDA** version has now been moved to **9.2**. 
The support for extended C++ lambdas in CUDA is not complete. The restrictions are discussed in the page:

https://docs.nvidia.com/cuda/archive/10.2/cuda-c-programming-guide/index.html#extended-lambda

Hydra3 can not wrap generic lambdas in host code. If this feature is necessary, use host-only uwrapped lambdas.

#### Function call interface

This is probably the most impacting change in this release, making **Hydra3** series backward incompatible with the previous series.

1. New interface for calling functors and lambdas. 

    a) Extensive statically-bound named parameter idiom support. This new idiom for specification of function call interfaces makes the definition callable objects in **Hydra3** much more safe, straight forward, transparent and user friendly, without compromise performance. In many cases enhancing performance, indeed. From **Hydra3**, users will be able to define new types, with *ad hoc* names wrapping around primary types, using the macro ```declarg(NewVar, Type)```. 
    These new types are searched in compile time to bind the function call, if the type
is not found a compile error is emitted, avoiding the generation of invalid or error prone code.
See how it works:

    ```cpp
    ...
    #include <hydra/functions/Gaussian.h>
    ...
    
    declarg(Angle, double)
    
    int main(int argv, char** argc)
    {
    ...
    
    auto gauss = hydra::Gaussian<Angle>(mean, signa);
    ...
    }
    ```
    
   in the previous code snippet, wherever the object `gauss` is called, if the argument consists of one or tuples, which are the entry type of all             multidimensional dataset classes in Hydra, the `Angle`type identifier will be searched among the elements, if not found, code will not compile. If the argument is a non-tuple type, conversion will be tried.  Multidimensional datasets can be defined using named parameters like in the snippet below:

    ```cpp
    
    ...
    #include <hydra/multivector.h>
    ...
     
    declarg(X, double)
    declarg(Y, double)
    declarg(Z, double)
    
    int main(int argv, char** argc)
    {
    //3D device buffer
    hydra::multivector< hydra::tuple<X,Y,Z>,  hydra::device::sys_t> data(nentries);
    
    ...
    
    for(auto x: hydra::column<X>(data)
        std::cout << x << std::endl;
    }
    ```       
    
   b) Functors: as usual, it should derive from ``hydra::BaseFunctor``, defined in ``hydra/Function.h``, but now the must inform their argument type, signature and number of parameters (``hydra::Parameter``) at template instantiation time. It is also necessary to implement the ``ResultType Evaluate(ArgType...)`` method. Default constructors should be deleted, non-default and copy constructors, as well as assignments operators should be implemented as well. See how this works for `hydra::Gaussian`:
    
    ```cpp
    
    //implementations omited, for complete details
    //see: hydra/functions/Gaussian.h

    template<typename ArgType, typename Signature=double(ArgType) >
    class Gaussian: public BaseFunctor<Gaussian<ArgType>, Signature, 2>
    {
    
    public:

        Gaussian()=delete;

        Gaussian(Parameter const& mean, Parameter const& sigma );

        __hydra_host__ __hydra_device__
        Gaussian(Gaussian<ArgType> const& other );

        __hydra_host__ __hydra_device__
        Gaussian<ArgType>& operator=(Gaussian<ArgType> const& other );

        __hydra_host__ __hydra_device__
        inline double Evaluate(ArgType x)  const;
    
    };
    ```
    
   c) Lambdas: Support for lambdas was updated to adhere the new interface. The new interface is implemented in `hydra/Lambda.h
    
    ```cpp    
    ...
    #include <hydra/multivector.h>
    #include <hydra/Lambda.h>
    ...

    declarg(X, double)
    declarg(Y, double)
    declarg(Z, double)

    int main(int argv, char** argc)
    {
        //3D device buffer
        hydra::multivector< hydra::tuple<X,Y,Z>,  hydra::device::sys_t> data(nentries);

        //Lambda
        auto printer = hydra::wrap_lambda( []__hydra_dual__(X x, Y y){

           print("x = %f y = %f", x(), y());

        } );

        for(auto entry: data) printer(entry);           
    }
    
    ```
    
#### Random number generation

1. Support for analytical pseudo-random number generation (APRNG) added for many functors added via `hydra::Distribution<FunctorType>` specializations (see example `example/random/basic_distributions.inl`).
2. Parallel filling of containers with random numbers (see example `example/random/fill_basic_distributions.inl`). In particular, there are some convenience functions in order to deploy in a generic and simple way the parallel filling of containers transparently, independently of the back-end. For a given instance of the functor of interest, the framework is informed of the presence of the APRNG method in compile time. If the APRNG 
is not found a compile error is emitted, informing and suggesting the user to use the `hydra::sample` interface, which implements a different filling strategy. The facility is decoupled from the underlying PRNG engine, in order to be compatible with the current pseudo-random engines already imlemented in Hydra, and future algorithms that will be made available over the time. If the user needs to use a specific PRNG engine, its type can be passed as template parameter to the convenience function, otherwise the `hydra_thrust::default_random_engine` is used. As an example of filling of containers with random numbers see the snippet below:

    ```cpp
    ...
    #include <hydra/functions/Gaussian.h>
    ...
    
    declarg(Xvar, double)
    
    int main(int argv, char** argc)
    {
    ...
    
    auto gauss  = hydra::Gaussian<Xvar>(mean, signa);
    
    auto data_d = hydra::device::vector<Xvar>(nentries);

    hydra::fill_random(data_d , gauss);

    ...
    }
    ```
The filling functions can be called also with a specific backend policy and with iterators instead of the whole container. The seed used by the PRNG engine can be also passed to the function as last parameter. The collection of all the convenience functions can be found in `hydra/RandomFill.h`.


#### Phase-space generation

1. Updated `hydra::Decays` container for supporting named variable idiom.
2. Changes in `hydra::PhaseSpace`and `hydra::Decays`.
3. hydra::Chain not supported any more.
4. New `Meld(...)` method in `hydra::Decays` for building mixed datasets and decay chains. 
5. Re-implemented logics for generation of events and associated weights.

#### Data fitting

1. Added support to multi-layered simultaneous fit of different models, over different datasets, deploying different parallelization strategies for each model. No categorization of the dataset is needed, but just to set up preliminarly the different component FCNs, that can be optimized in isolation or in the context of the simultaneous FCN. Simultaneous FCNs can be created via direct instantiation or using the convenience function `hydra::make_simultaneous_fcn(...)`, as shown in the following snippet.

    ```cpp
    ...
    #include <hydra/LogLikelihoodFCN.h>
    ...
    
    int main(int argv, char** argc)
    {
     ...
     //=====================================================================================
     //                                                           +----< fcn(model-x) 
     //                           +----< simultaneous fcn 1 ----- |
     //                           |                               +----< fcn(model-y)  
     //   simultaneous fcn   <----+
     //                           |                               +----< fcn(model-w)
     //                           +----< simultaneous fcn 2 ------|
     //                           |                               +----< fcn(model-z) 
     //                           +----< fcn(model-v)
     //=====================================================================================        
     auto fcnX    = hydra::make_loglikehood_fcn(modelX, dataX);
     auto fcnY    = hydra::make_loglikehood_fcn(modelY, dataY);
     auto fcnW    = hydra::make_loglikehood_fcn(modelY, dataY);
     auto fcnZ    = hydra::make_loglikehood_fcn(modelZ, dataZ);    
     auto fcnV    = hydra::make_loglikehood_fcn(modelv, datav);
     
     auto sim_fcn1 = hydra::make_simultaneous_fcn(fcnx, fcny);
     auto sim_fcn2 = hydra::make_simultaneous_fcn(fcnw, fcnz);
     auto sim_fcn  = hydra::make_simultaneous_fcn(sim_fcn1, sim_fcn2, fcnV);
     ...
    }
    ```
    Moreover, the generic interface allows to build up a simultaneous FCN object by composing usual FCNs and simultaneous FCNs. An example of such new method can be found in `examples/fit/simultaneous_fit.inl`.

2. Fitting of convoluted PDFs.

#### General
Many issues solved and bugs fixed across the tree:

    1. https://github.com/MultithreadCorner/Hydra/issues/91#issue-631032116
    2. https://github.com/MultithreadCorner/Hydra/issues/90
    3. https://github.com/MultithreadCorner/Hydra/pull/89
    4. https://github.com/MultithreadCorner/Hydra/issues/87
    5. https://github.com/MultithreadCorner/Hydra/issues/86
    6. https://github.com/MultithreadCorner/Hydra/issues/82
    7. https://github.com/MultithreadCorner/Hydra/issues/77
   
 and many others. 
 
-------------------------

### Hydra 2.6.0

This is the last release from series 2.x.x.

-------------------------

### Hydra 2.5.0

1. **Eigen** is not being distributed with **Hydra** anymore. **Eigen** will remain an dependency for foreseeable future.
2. New facility to update **Thrust** and **CUB**. New namespaces ```hydra::hydra_thrust``` and ```hydra::hydra_cub``` defined.
3. New lazy implementation of ```hydra::Splot```.
4. New ```hydra::PhaseSpace``` interface, with constructors taking the mass of the mother particle as well. Ctors also protected against inconsistent arguments via exception throwing. 
5. New algorithm ```hydra::sobol``` support up to 3667 dimensions
6. Re-implementation of the impacted examples. 
7. Many bug fixes across the tree...

### Hydra 2.4.1 (probably incomplete)

1. The main change is this release is the update of Thrust instance distributed with Hydra to the version 1.9.6, which enabled the support for CUDA 10.1 and hopefuly higher
2. Range semantics implemented in Decays::Unweight methods
3. Fix CLANG discovery on Apple platform (not officially supported yet)
4. Many bug fixes across the tree...

### Hydra 2.3.1  (probably incomplete)

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
