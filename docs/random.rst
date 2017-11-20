Random number generation and PDF sampling
=========================================

Generation of random numbers and sampling of multidimensional PDFs is supported in Hydra through
the class ``hydra::Random<typename Engine>``, where ``Engine`` is the random number generator engine. There are four random number engines available 

1. ``hydra::minstd_rand0``: implements a version of the Minimal Standard random number generation algorithm.
2. ``hydra::minstd_rand``: implements a version of the Minimal Standard random number generation algorithm.
3. ``hydra::ranlux24``: RANLUX level-3 random number generation algorithm.
4. ``hydra::ranlux48``:  RANLUX level-4 random number generation algorithm.
5. ``hydra::taus88``:  L'Ecuyer's 1996 three-component Tausworthe random number generator.

The default random number generation engine is ``hydra::minstd_rand0``.
This class provides methods that take iterators pointing to containers that will be filled with random numbers distributed according the requested distributions. If a explicit back-end policy is passed, the generation is parallelized in the corresponding back-end, otherwise the class will process the random number generation in the back-end the containers is allocated. The 

Sampling basic distributions
----------------------------

``hydra::Random`` defines four methods to generate predefined one-dimensional.
These methods are summarized below, where begin and end are iterators pointing to the 
range that will filled with random numbers. The other parameters represent the standard definitions: 

	1. ``hydra::Random::Gauss( mean, sigma, begin, end)`` for Gaussian distribution. 
	2. ``hydra::Random::Exp(tau, begin, end)`` for exponential distribution.
	3. ``hydra::Random::Uniform(min, max, begin, end)`` for an uniform distribution.
	4. ``hydra::Random::BreitWigner(mean, width, begin, end)`` for non-relativistic Breit-Wigner distribution.
	
The example below show how to use these methods

.. code-block:: cpp
	
	#include <hydra/device/System.h>
	#include <hydra/Random.h>
	
	...

	hydra::Random<>	Generator(4598635);
	hydra::device::vector<double>  data(1e6);

	//uniform distribution in the interval [-5,5]
	Generator.Uniform(-5.0, 5.0, data.begin(), data.end());
	//Gaussian distribion with mean=0 and sigma =1
	Generator.Gauss(0.0, 1.0, data.begin(), data.end());
	//exponential distribion with tau=1
	Generator.Exp(1.0, data.begin(), data.end());
	//Breit-Wigner with mean 2.0 width 0.2 
	Generator.BreitWigner(2.0, 0.2, data_d.begin(), data_d.end());
		

Multidimensional PDF sampling
-----------------------------


The class ``hydra::Random`` supports the sampling of multidimensional probability density functions (PDF) through the method ``hydra::Random::Sample(begin, end, min, max, functor)``, where ``min`` and ``max`` are static arrays or ``std::array`` objects representing the limits of the multidimensional region. ``functor`` is a Hydra functor representing the PDF. 

The PDFs are sampled using the accept-reject method. The ``hydra::Random::Sample`` returns a ``hydra::GenericRange`` object pointing to a range filled with the sampled numbers. The PDF sampling
is processed filling the container with random numbers and reordering it to reproduce the shape 
of the PDF, no memory reallocation is performed. The range returned by the method points to a sub-set of the original container, the size of this range depends on the efficiency of the accept-reject for the given PDF.


.. code-block:: cpp
	
	#include <hydra/device/System.h>
	#include <hydra/Random.h>
	#include <hydra/FunctionWrapper.h>

	...
	
	double mean   =  0.0;
	double sigma  =  1.0;

	auto gaussian = hydra::wrap_lambda( 
		[=] __host__ __device__ (unsigned int n,double* x ){

			double g = 1.0;

			for(size_t i=0; i<3; i++){
				double m2 = (x[i] - mean )*(x[i] - mean );
				double s2 = sigma*sigma;
				g *= exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));
			}

			return g;
		}
	);

	std::array<double, 3> max{ 6.0,  6.0,  6.0};
	std::array<double, 3> min{-6.0, -6.0, -6.0};

	hydra::multiarray<3, double, hydra::device::sys_t> data;
	auto range = Generator.Sample(data.begin(),  data.end(), min, max, gaussian);


