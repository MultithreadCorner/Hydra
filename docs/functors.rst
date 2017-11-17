Functors and C++11 lambdas.
==========================


The user's code is passed to Hydra's algorithms through functors and C++11 lambda functions.
Hydra adds type information and functionality to functors and lambdas using CRTP idiom.
Functors and lambdas are  entities and are not attached to a specific back-end.
The signatures conventions adopted for functors and lambdas as well as the added functionality
will be discussed in the following lines.

Functors
--------

In C++, a functor, sometimes also referred as a function object, is any class or structure that overloads the function call operator ``òperator()``. In Hydra, all functors derives from the 
class template ``hydra::BaseFunctor<Functor, ReturnType, NParameters>``. The template parameters
are described below:

	* ``Functor`` : the type of the functor. 
	* ``ReturnType``: the type of the functor.
	* ``NParameters``: the number of parameters the functor has. 

The user needs only to implement the function ``Evaluate()`` and Hydra will take of implementing the function call operator. The signature of ``Evaluate()`` depends on the type of data that will be passed. There are two possibilities:

	1. The functor is supposed to take as arguments data with the same type. In this case 
	the signature of the function call operator will be 
	
	.. code-block:: cpp
	 :linenos:

		template<typename T> 
		__host__ __device__ 
		ReturnType Evaluate(unsigned int n , T* x);
	
	where ``T`` is the data type, ``n`` the number of arguments and ``x`` a pointer to an array of arguments. The symbols ``__host__ __device__`` are the necessary to make the functor callable on host and device memory spaces. 
	
	2. The functor is supposed to take as arguments data with different types. In this case the signature of the function call operator will be 
	
	.. code-block:: cpp 
		:linenos:
	
		template<typename T> 
		__host__ __device__ 
		ReturnType Evaluate(T& x);
	
	where T is the data type, in this case a ``hydra::tuple`` of arguments.

The parameters are represented by the ``hydra::Parameter``. The parameters can be named, store maximum and minimum values and error. The objects of the class ``hydra::Parameter`` can be instantiated using named field idiom or field list idiom, like this    



.. code-block:: cpp 
		
	#include <hydra::Parameters.h>
	#include <string>
	

    std::string p1_name("p1");
	auto p1 = hydra::Parameter::Create().Name(p1).Value(0.0).Limits(-1.0, 1.0).Error(0.01);

	std::string p2_name("p2");
	auto p2 = hydra::Parameter(p1,0.0,0.001,-1.0, 1.0);

Hydra does not check the name of the parameters in any way. It is up to the user to care about the contexts where parameters can have or not the same name. 
The parameters of a functor are accessible via the ``_par[]`` subscript operator or invoking the ``GetParameter(unsigned int i)`` member function 

As an example, let's consider the Gaussian function with mean :math:`\mu` and sigma :math:`\sigma`


.. math:: f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}


and suppose the corresponding functor will take as arguments data with same type and evaluate the Gaussian on the first argument. The code in Hydra would look like this


.. code-block:: cpp
	:linenos: 


	#include <hydra::Parameters.h>
	#include <string>
	
	...

    
	struct Gaussian: public <Gaussian, double, 2>
	{

		// delete the default constructor.
		// user always have to inform mean and and sigma 
		Gaussian()= delete;

		//constructor
		Gaussian(hydra::Parameter mean, hydra::Parameter sigma):
		hydra::BaseFunctor({mean, sigma}) 
		{}

		template<typename T>
		__host__ __device__
		double Evaluate(unsigned int n , T* x)
		{
 
			double mean  = _par[0];
			double sigma = _par[1];

			double x2 = (x[0]-mean)*(x[0]-mean);
			double s2 = sigma*sigma;

			return exp(- x2/(2.0*s2 ))/( sqrt(2.0*s2*PI));
		}

	};

	...

	std::string mean_name("mean");
	auto m = hydra::Parameter::Create().Name(mean).Value(0.0).Limits(-1.0, 1.0).Error(0.01);

	std::string sigma_name("sigma");
	auto s = hydra::Parameter::Create().Name(sigma).Value(1.0).Limits(0.01, 5.0).Error(0.01);

	Gaussian gauss(m, s);
	

The Gaussian implementation can be generalized to allow the functor to operate over any type of arguments overloading the `Evaluate()` method and adding a template parameter 
to represent which argument the functor will use to evaluate the Gaussian. Se this implementation below 


.. code-block:: cpp
	:linenos:

	#include <hydra::Parameters.h>
	#include <string>

	template<unsigned int Index>
	struct Gaussian: public <Gaussian<Index>, double, 2>
	{

		// delete the default constructor.
		// user always have to inform mean and and sigma 
		Gaussian()= delete;

		//constructor
		Gaussian(hydra::Parameter mean, hydra::Parameter sigma):
		hydra::BaseFunctor({mean, sigma}) 
		{}

		template<typename T>
		__host__ __device__
		double Evaluate(unsigned int n , T* x)
		{
 
			double mean  = _par[0];
			double sigma = _par[1];

			double x2 = (x[Index]-mean)*(x[Index]-mean);
			double s2 = sigma*sigma;

			return exp(- x2/(2.0*s2 ))/( sqrt(2.0*s2*PI));
		}

		template<typename T>
		__host__ __device__
		double Evaluate(T x)
		{
 
			double mean  = _par[0];
			double sigma = _par[1];

			double x2 = (hydra::get<Index>(x)-mean)*( hydra::get<Index>(x)-mean);
			double s2 = sigma*sigma;

			return exp(- x2/(2.0*s2 ))/( sqrt(2.0*s2*PI));
		}


	};

	...
    
	std::string mean_name("mean");
	auto m = hydra::Parameter::Create().Name(mean).Value(0.0).Limits(-1.0, 1.0).Error(0.01);

	std::string sigma_name("sigma");
	auto s = hydra::Parameter::Create().Name(sigma).Value(1.0).Limits(0.01, 5.0).Error(0.01);

	Gaussian<0> gauss1(m, s);
	Gaussian<2> gauss2(m, s);
	
	

 
