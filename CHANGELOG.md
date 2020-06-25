## CHANGE LOG

### Hydra 3.0.0

It is the first release of the longly  waited 3 series. Overall, this release is expected to be faster
or at least to have similar performance to the previous releases. 
There are many bug fixes and other changes across. The most significant are summarized below:

#### Function call interface

This is probably the most impacting change in this release, making **Hydra3** series backward incompatible with the previous series.

1. New interface for calling functors and lambdas. 

    a) Extensive statically-bound named parameter idiom support. This new idiom for specification of function call interfaces makes the definition callable objects in **Hydra3** much more safe, straight forward, transparent and user friendly, without compromise performance. In many cases enhancing performance, indeed. From **Hydra3**, users will be able to define new types, with *ad hoc* names wrapping around primary types, using the macro ```declvar(NewVar, Type)```. 
    These new types are searched in compile time to bind the function call, if the type
is not found a compile error is emitted, avoiding the generation of invalid or error prone code.
See how it works:

    ```cpp
    ...
    #include <hydra/functions/Gaussian.h>
    ...
    
    declvar(Angle, double)
    
    int main(int argv, char** argc)
    {
    ...
    
    auto gauss = hydra::Gaussian<Angle>(mean, signa);
    ...
    }
    ```  
 in the previous code snippet, wherever the object `gauss` is called, if the argument consists of one or tuples, which are the entry type of all multidimensional dataset classes in Hydra, the `Angle`type identifier will be searched among the elements, if not found, code will not compile. If the argument is a non-tuple type, conversion will be tried.  Multidimensional datasets can be defined using named parameters like in the snippet below:
 
    ```cpp
    ...
    #include <hydra/multivector.h>
    ...

    declvar(X, double)
    declvar(Y, double)
    declvar(Z, double)

    int main(int argv, char** argc)
    {
        //3D device buffer
        hydra::multivector< hydra::tuple<X,Y,Z>,  hydra::device::sys_t> data(nentries);

        ...

        for(auto x: hydra::column<X>(data)
        {

        std::cout << x << std::endl;

        }
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
    c) Lambdas: Support for lambdas is updated for the new interface. The new interface is implemented in `hydra/Lambda.h
    
    ```cpp    
    ...
    #include <hydra/multivector.h>
    #include <hydra/Lambda.h>
    ...

    declvar(X, double)
    declvar(Y, double)
    declvar(Z, double)

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
