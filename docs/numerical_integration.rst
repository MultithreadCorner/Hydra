Numerical integration
=====================

Numerical integration of multidimensional functions is not easy. Numerical integration algorithms tend to be involved and resource hungry, since they demand a large number of evaluations of the integrand. Indeed, it is common to express the efficiency of a given algorithm in terms of the minimum required number of integrand evaluations to achieve given precision for the integral estimation. 

The best strategy to perform numerical integration largely depends on  the number of dimensions of the integration region and on the features of the integrand in this region. Given the detailed information about integrand behavior is usually not available, the numerical integration algorithms need to handle situations specializing routines based on broad properties of the integrand, like for example, the presence or not of narrow peaks, periodicity etc. Highly specialized algorithms, optimized to handle only a given class of problems tend to be more efficient, but on the other hand, such approaches are usually not applicable to even a slight different problems. In other words, on the field of numerical integration, flexibility usually comes at expenses of efficiency. The basics of numerical integration followed by comprehensive list of references on the subject can be found in the Wikipedia pages `<https://en.wikipedia.org/wiki/Numerical_integration>`_ and `<https://en.wikipedia.org/wiki/Monte_Carlo_integration>`_ .

Hydra provides a set of parallelized implementations for generic and popular algorithms to compute one- and multidimensional numerical integration. Hydra parallelizes the calls to the integrand in a collection of threads in requested back-end. The algorithms share the same basic call interface, manage resources using RAII idiom and estimate the integral and the associated error. Hydra also supports analytical integration, through functors. 


Gauss-Kronrod quadrature 
------------------------

.. seealso:: Good didactic introduction on Gauss-Kronrod quadrature can be found in the Wikipedia page `<https://en.wikipedia.org/wiki/Gauss-Kronrod_quadrature_formula>`_. 


The class ``hydra::GaussKronrodQuadrature<NRULE, NBIN, Backend>`` implements a non-adaptive procedure which divides the integration interval in ``NBIN`` sub-intervals and  applies to each sub-interval a fixed Gauss-Kronrod rule of order ``NRULE``. ``hydra::GaussKronrodQuadrature<NRULE, NBIN, Backend>`` allows the fast integration of smooth one-dimensional functions.
The code snippet below show how to use this quadrature to calculate the integral of a Gaussian function:


.. code-block:: cpp


	#include <hydra/GaussKronrodQuadrature.h>
	#include <hydra/FunctionWrapper.h>
	#include <hydra/device/System.h>

	...

	//integration region limits
	double  min  = -6.0;
	double  max =  6.0;

	//Gaussian parameters
	double mean  = 0.0;
	double sigma = 1.0;


	//wrap the lambda
	auto gaussian = hydra::wrap_lambda(
	[=] __host__ __device__ (unsigned int n, double* x ){

		double m2 = (x[0] - mean )*(x[0] - mean );
		double s2 = sigma*sigma;
		double f = exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));

		return f;
	} );

	...

	// 61- degree quadrature
	hydra::GaussKronrodQuadrature<61,100, hydra::device::sys_t> GKQ61_d(min,  max);

	auto result = GKQ61_d.Integrate(gaussian);

	std::cout << "Result: " << result.first << "  +-  " << result.second <<std::endl



Self-Adaptive Gauss-Kronrod quadrature 
--------------------------------------

.. seealso:: Good didactic introduction on Gauss-Kronrod quadrature can be found in the Wikipedia page `<https://en.wikipedia.org/wiki/Gauss-Kronrod_quadrature_formula>`_. 


The class ``hydra::GaussKronrodAdaptiveQuadrature<NRULE, NBIN, Backend>`` implements a self-adaptive algorithm  which initially divides the integration interval in ``NBIN`` sub-intervals and  applies to each sub-interval a Gauss-Kronrod rule of order ``NRULE``. The algorith selects the interval with larger relative error in the integral estimation and re-applies the procedure. The algorithme keeps performing this loop until the integram estimation reaches the requested maximum error level. 

``hydra::GaussKronrodQAdaptiveuadrature<NRULE, NBIN, Backend>`` performs less calls to the integrand and is best suitable for very featured and expensive functions. The code snippet below show how to use this quadrature to calculate the integral of a Gaussian function:


.. code-block:: cpp


	#include <hydra/GaussKronrodAdaptiveQuadrature.h>
	#include <hydra/FunctionWrapper.h>
	#include <hydra/device/System.h>

	...

	//integration region limits
	double  min  = -6.0;
	double  max  =  6.0;
    double max_error = 1e-6;

	//Gaussian parameters
	double mean  = 0.0;
	double sigma = 1.0;


	//wrap the lambda
	auto gaussian = hydra::wrap_lambda(
	[=] __host__ __device__ (unsigned int n, double* x ){

		double m2 = (x[0] - mean )*(x[0] - mean );
		double s2 = sigma*sigma;
		double f = exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));

		return f;
	} );

	...

	// 61- degree quadrature
	hydra::GaussKronrodQuadrature<61,10, hydra::device::sys_t> GKQ61(min,  max, max_error);

	auto result = GKQ61.Integrate(gaussian);

	std::cout << "Result: " << result.first << "  +-  " << result.second <<std::endl



Genz-Malik multidimensional quadrature 
--------------------------------------

This method implements a polynomial interpolatory rule of degree 7, which integrates
exactly all monomials :math:`{x_1}^{k_1}, {x_2}^{k_2} . . . {x_n}^{k_d}` with :math:`\sum k_i \leq 7` and fails to integrate exactly at least one monomial of degree 8. In the [Genz-Malik]_ multidimensional quadrature, all integration nodes are inside integration domain and
:math:`2^d + 2d^2 + 2d + 1` integrand evaluations are required to integrate a function in a 
rectangular hypercube with d dimensions. Due the fast increase in the number of evaluations as a function of the dimension, this method is most advantageous for problems with d < 10 and is superseded for high-dimensional integrals by Monte Carlo based methods.
A degree 5 rule embedded in the degree 7 rule is used for error
estimation, in a such way that no additional integrand evaluations are necessary.

The class template ``hydra::GenzMalikQuadrature<N, BackendPolicy >`` implements a static version of Genz-Malik multidimensional quadrature. This version divides
the ``N``dimensional integration region in a series of sub-regions, according the configuration, passed by the user and applies the rule to each sub-region. 


The code snippet below shows to use the ``hydra::GenzMalikQuadrature<N, BackendPolicy >``
class to integrate a five-dimensional Gaussian distribution. In this example each 
dimension is divided in 10 segments, resulting in :math:`10^5` sub-regions.

.. code-block:: cpp
	
	#include <hydra/GaussKronrodAdaptiveQuadrature.h>
	#include <hydra/FunctionWrapper.h>
	#include <hydra/device/System.h>

	...

	//number of dimensions (user can change it)
	constexpr size_t N = 5;

	//integration region limits
	double  min[N];
	double  max[N];
	size_t  grid[N];

	//5D Gaussian parameters
	double mean  = 0.0;
	double sigma = 1.0;

	//set Gaussian parameters and
	//integration region limits
	for(size_t i=0; i< N; i++){
		min[i]   = -6.0;
		max[i]   =  6.0;
		grid[10] =  10;
	}

	//wrap the lambda
	auto gaussian = hydra::wrap_lambda( [=] __host__ __device__ (unsigned int n, double* x ){

		double g = 1.0;
		double f = 0.0;

		for(size_t i=0; i<N; i++){

			double m2 = (x[i] - mean )*(x[i] - mean );
			double s2 = sigma*sigma;
			f = exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));
			g *= f;
		}

		return g;
	});

	hydra::GenzMalikQuadrature<N, hydra::device::sys_t> GMQ(min, max, grid);

	auto result = GMQ.Integrate(gaussian);

	std::cout << "Result: " << result.first << "  +-  " << result.second <<std::endl


Plain Monte Carlo
-----------------

The plain Monte Carlo algorithm samples points randomly from the integration region to estimate the integral and its error. Using this algorithm the estimate of the integral E(f; N) for N randomly distributed points x_i is given by,

.. math::

	E(f; N) = V<f> = (V / N) \sum_i^N f(x_i)


where V is the volume of the integration region. The error on this estimate :math:`\sigma(E;N)` is calculated from the estimated variance of the mean,

.. math::

	\sigma^2 (E; N) = (V^2 / N^2) \sum_i^N (f(x_i) -  <f>)^2


For large N this variance decreases asymptotically as :math:`Var(f)/N`, where :math:`Var(f)` is the true variance of the function over the integration region. The error estimate itself should decrease as :math:`\sigma(f)/\sqrt{N}`, which implies that to reduce the error by a factor of 10, a 100-fold increase in the number of sample points is required.

Hydra implements the plain Monte Carlo method in the class ``hydra::Plain<N, BackendPolicy>``, where N is the number of dimensions and ``BackendPolicy`` is the back-end to parallelize the calculation.

The following code snippet shows to use the ``hydra::Plain<N, BackendPolicy >``
class to integrate a five-dimensional Gaussian distribution performing 100

.. code-block:: cpp
	
	#include <hydra/FunctionWrapper.h>
	#include <hydra/device/System.h>
	#include <hydra/Plain.h>

	...

	//number of dimensions (user can change it)
	constexpr size_t N = 5;

	//integration region limits
	double  min[N];
	double  max[N];
	size_t  ncalls = 1e6;

	//5D Gaussian parameters
	double mean  = 0.0;
	double sigma = 1.0;

	//set Gaussian parameters and
	//integration region limits
	for(size_t i=0; i< N; i++){
		min[i]   = -6.0;
		max[i]   =  6.0;
	}

	//wrap the lambda
	auto gaussian = hydra::wrap_lambda( [=] __host__ __device__ (unsigned int n, double* x ){

		double g = 1.0;
		double f = 0.0;

		for(size_t i=0; i<N; i++){

			double m2 = (x[i] - mean )*(x[i] - mean );
			double s2 = sigma*sigma;
			f = exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));
			g *= f;
		}

		return g;
	});

	hydra::Plain<N, hydra::device::sys_t> PlainMC(min, max, ncalls);

	auto result = PlainMC.Integrate(gaussian);

	std::cout << "Result: " << result.first << "  +-  " << result.second <<std::endl


Self-adaptive importance sampling (Vegas)
------------------------------------------

 
.. note:: 

 	from GSL's Manual, chapter 'Monte Carlo integration' `<https://www.gnu.org/software/gsl/manual/html_node/VEGAS.html>`_ : 

		The VEGAS algorithm of [Lepage]_ is based on importance sampling. It samples points from the probability distribution described by the function :math:`|f|`, so that the points are concentrated in the regions that make the largest contribution to the integral.

		In general, if the Monte Carlo integral of f is sampled with points distributed according to a probability distribution described by the function g, we obtain an estimate  :math:`E_g(f; N)`,

		.. math::

		 	E_g(f; N) = E(f/g; N)

		with a corresponding variance,
		
		.. math::
		
			{Var}_g(f; N) = Var(f/g; N).

		If the probability distribution is chosen as :math:`g = |f|/I(|f|)` then it can be shown that the variance {Var}_g(f; N) vanishes, and the error in the estimate will be zero. In practice it is not possible to sample from the exact distribution g for an arbitrary function, so importance sampling algorithms aim to produce efficient approximations to the desired distribution.

		The VEGAS algorithm approximates the exact distribution by making a number of passes over the integration region while histogramming the function f. Each histogram is used to define a sampling distribution for the next pass. Asymptotically this procedure converges to the desired distribution. In order to avoid the number of histogram bins growing like K^d the probability distribution is approximated by a separable function: :math:`g(x_1, x_2, ...) = g_1(x_1) g_2(x_2) ...` so that the number of bins required is only :math:`K_d`. This is equivalent to locating the peaks of the function from the projections of the integrand onto the coordinate axes. The efficiency of VEGAS depends on the validity of this assumption. It is most efficient when the peaks of the integrand are well-localized. If an integrand can be rewritten in a form which is approximately separable this will increase the efficiency of integration with VEGAS.
		
		...


The implementation of VEGAS in Hydra parallelizes the Monte Carlo generation, the function calls and the computing of the result of each iteration. The algorithm is implemented in the
``hydra::Vegas<N,  BackendPolicy>``. The auxiliary class ``hydra::VegasState<N,  BackendPolicy>`` manages the resources and configuration necessary to perform the integration. The code snippet below shows how to use the VEGAS algorithm to integrate five-dimensional Gaussian distribution:

 .. code-block:: cpp

	#include <hydra/Vegas.h>
	#include <hydra/FunctionWrapper.h>
	#include <hydra/device/System.h>
	 
	...

	//number of dimensions (user can change it)
	constexpr size_t N = 5;

	//integration region limits
	double  min[N];
	double  max[N];
	size_t  ncalls = 1e5;

	//5D Gaussian parameters
	double mean  = 0.0;
	double sigma = 1.0;

	//set Gaussian parameters and
	//integration region limits
	for(size_t i=0; i< N; i++){
		min[i]   = -6.0;
		max[i]   =  6.0;
	}

	//wrap the lambda
	auto gaussian = hydra::wrap_lambda(
		[=] __host__ __device__ (unsigned int n, double* x ){

			double g = 1.0;
			double f = 0.0;

			for(size_t i=0; i<N; i++){

				double m2 = (x[i] - mean )*(x[i] - mean );
				double s2 = sigma*sigma;
				f = exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));
				g *= f;
			}

			return g;
		}
	);

	//vegas integrator
	hydra::Vegas<N,  hydra::device::sys_t> Vegas(min, max, ncalls);

	//configuration
	Vegas.GetState().SetVerbose(-2);
	Vegas.GetState().SetAlpha(1.5);
	Vegas.GetState().SetIterations( iterations );
	Vegas.GetState().SetUseRelativeError(1);
	Vegas.GetState().SetMaxError( max_error );
	Vegas.GetState().SetCalls( calls );
	Vegas.GetState().SetTrainingCalls( calls/10 );
	Vegas.GetState().SetTrainingIterations(2);

	auto result = Vegas_d.Integrate(gaussian);
	std::cout << "Result: " << result.first << "  +-  " << result.second <<std::endl

Implementing analytical integration
------------------------------------

Hydra supports analysical integration as well. To integrate functions analytically the user needs to implement the integral formula in a suitable functor ``Functor`` deriving from the class 
``hydra::Integrator<Functor>``. Analytical integration is not parallelized. 

