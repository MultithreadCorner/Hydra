Functors and C++11 lambdas.
===========================


The user's code is passed to Hydra's algorithms through functors and C++11 lambda functions.
Hydra adds type information and functionality to functors and lambdas using CRTP idiom.
Functors and lambdas are  entities not attached to a specific back-end.
The signatures conventions adopted for functors and lambdas as well as the added functionality
will be discussed in the following lines.

Functors
--------

In C++, a functor, sometimes also referred as a function object, is any class or structure that overloads the function call operator ``òperator()(Args ...x)``. In Hydra, all functors derives from the class template ``hydra::BaseFunctor<Functor, ReturnType, NParameters>``. The template parameters are described below:

	* ``Functor`` : the type of the functor. 
	* ``ReturnType``: the type returned by the functor.
	* ``NParameters``: the number of parameters the functor has. 

The user needs only to implement the method ``Evaluate(...)`` and Hydra will take care of implementing the function call operator. The signature of ``Evaluate(...)`` depends on the type of data that will be passed. There are two possibilities:

	1. The functor is supposed to take as arguments (one-)multidimensional data with the same type. In this case the signature of the ``Evaluate(...)`` method will be 
	
	.. code-block:: cpp

		template<typename T> 
		__host__ __device__ 
		ReturnType Evaluate(unsigned int n , T* x);
	
	where ``T`` is the data type, ``n`` the number of arguments and ``x`` a pointer to an array of arguments. The symbols ``__host__ __device__`` are the necessary to make the functor callable on host and device memory spaces. 
	
	2. The functor is supposed to take as arguments multidimensional data with different types, so data will be compacted in a ``hydra::tuple`` object. In this case the signature of the function call operator will be 
	
	.. code-block:: cpp 
	
		template<typename T> 
		__host__ __device__ 
		ReturnType Evaluate(T& x);
	
	where T is the dataset entry type, in this case a ``hydra::tuple`` of arguments.

The parameters are represented by the ``hydra::Parameter``. The parameters can be named, store maximum and minimum values and error. The objects of the class ``hydra::Parameter`` can be instantiated using named field idiom or field list idiom, like this    


.. code-block:: cpp 

	#include <Parameter.h>
	#include <string>
	
	auto p1 = hydra::Parameter::Create().Name("p1").Value(0.0).Limits(-1.0, 1.0).Error(0.01);

	auto p2 = hydra::Parameter("p1",0.0,0.001,-1.0, 1.0);

Hydra does not check uniqueness of the name of the parameters at creation time in any way. It is up to the user to care about the contexts where parameters can have or not the same name. 
The parameters of a functor are accessible via the ``_par[]`` subscript operator or invoking the ``GetParameter(unsigned int i)`` and ``GetParameter(const char* name)`` functor member function. 

As an example, let's consider the Gaussian function with mean :math:`\mu` and sigma :math:`\sigma`


.. math:: f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}


and suppose the corresponding functor will take as arguments data with same type and evaluate the Gaussian on argument set by the template parameter ``ArgIndex``. 

The corresponding code in Hydra would look like this


.. code-block:: cpp
	:name: functor-example1

	#include <hydra/Parameters.h>
	#include <hydra/Function.h>
	    
	template<unsigned int ArgIndex=0>
	class Gaussian: public BaseFunctor<Gaussian<ArgIndex>, double, 2> {

	using BaseFunctor<Gaussian<ArgIndex>, double, 2>::_par;

	public:

	Gaussian()=delete;

	Gaussian(Parameter const& mean, Parameter const& sigma ):
	BaseFunctor<Gaussian<ArgIndex>, double, 2>({mean, sigma})
	{}

	__host__ __device__
	Gaussian(Gaussian<ArgIndex> const& other ):
	BaseFunctor<Gaussian<ArgIndex>, double,2>(other)
	{}

	__host__ __device__
	Gaussian<ArgIndex>&
	operator=(Gaussian<ArgIndex> const& other ){
	
	if(this==&other) return  *this;
	BaseFunctor<Gaussian<ArgIndex>,double, 2>::operator=(other);
	return  *this;
	
	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(unsigned int, T*x)  const	{
	
	double m2 = (x[ArgIndex] - _par[0])*(x[ArgIndex] - _par[0] );
	double s2 = _par[1]*_par[1];
	
	return exp(-m2/(2.0 * s2 ));

	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(T x)  const {

	double m2 = ( get<ArgIndex>(x) - _par[0])*(get<ArgIndex>(x) - _par[0] );
	double s2 = _par[1]*_par[1];
	
	return exp(-m2/(2.0 * s2 ));

	}

	};

	...

	auto m = hydra::Parameter::Create().Name("mean").Value(0.0).Limits(-1.0, 1.0).Error(0.01);

	auto s = hydra::Parameter::Create().Name("sigma").Value(1.0).Limits(0.01, 5.0).Error(0.01);

	Gaussian gauss(m, s);
	
	double args_single(1.0);
	hydra::tuple<int, double> args_tuple{0, 1.0};
	double args_array[2]{0.0, 1.0};

	// the following calls produces the same results
	std::cout	<< gauss(args_single)    << " " 
	       		<< gauss1(args_tuple) 	 << " "
				<< gauss1(2, args_array) << std::endl;   
	

Actually, Hydra users will rarely call functors directly. Functors are used to encapsulate user's
code that will be called in parallelized calculations by the Hydra algorithms in multi-threaded CPU and GPU environments. **It is user's responsibility care about race conditions and other problems bad coded functors can cause. It is strongly advised to avoid dynamic memory allocation inside functors.**   


C++11 Lambdas
-------------

Hydra fully supports C++11 lambdas. Before to pass C++11 lambdas to Hydra's algorithms, users need to wrap it into a suitable Hydra object. This is done invoking the function template 
``hydra::wrap_lambda()``.

As well as for functors, the signature of the lambda function depends on the type of data that will be passed. There are two possibilities:

	1. The functor is supposed to take as arguments data with the same type. In this case 
	the signature of the function call operator will be 
	
	.. code-block:: cpp

		[=]__host__ __device__(unsigned n, T* x){
		 //implementation goes here 
		};
	
	where ``T`` is the data type, ``n`` the number of arguments and ``x`` a pointer to an array of arguments. The symbols ``__host__ __device__`` are the necessary to make the lambda callable on host and device memory spaces. 
	
	2. The functor is supposed to take as arguments data with different types. In this case the signature of the function call operator will be 
	
	.. code-block:: cpp 
	
		[=]__host__ __device__(T x){
		 //implementation goes here 
		};
	
	where T is the data type, in this case a ``hydra::tuple`` of arguments.

Hydra can also handle "parametric lambdas". Parametric lambdas are wrapped C++11 lambdas that can hold named parameters (``hydra::Parameters`` objecs). 
The signatures for parametric lambdas are:


	1. The functor is supposed to take as arguments data with the same type. In this case 
	the signature of the function call operator will be 
	
	.. code-block:: cpp

		[=]__host__ __device__(unsigned int np, hydra::Parameters* p, unsigned na, T* args)
		{
		 //implementation goes here 
		};
	
	where ``nparams`` is the number of parameters, ``params`` is a pointer to the array of parameters, ``T`` is the data type, ``nargs`` the number of arguments and ``args`` a pointer to the array of arguments. The symbols ``__host__ __device__`` are the necessary to make the lambda callable on host and device memory spaces. 
	
	2. The functor is supposed to take as arguments data with different types. In this case the signature of the function call operator will be 
	
	.. code-block:: cpp 
	
		[=]__host__ __device__(unsigned int nparams, hydra::Parameters* params, T args)
		{
		 //implementation goes here 
		};
	
	where ``nparams`` is the number of parameters, ``params`` is a pointer to the array of parameters and ``T`` is the data type, in this case, a ``hydra::tuple`` of arguments.

The following example shows how to wrap a lambda to calculate a Gaussian function capturing the mean and sigma from the lambda's enclosing scope:


.. code-block:: cpp
	:name: lambda-example1

	#include <hydra/FunctorWrapper.h>

	...

	double mean  = 0.0;
	double sigma = 1.0;

	auto raw_gaussian = [=] __host__ __device__ (unsigned int nargs, double* args){

		double m2 = (x[0] - mean )*(x[0] - mean );
		double s2 = sigma*sigma;
		
		return exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));

	};

	auto wrapped_gaussian = hydra::wrap_lambda(raw_gaussian);


In the :ref:`previous example <lambda-example1>` the mean and the sigma of the Gaussian can not be changed once the lambda is instantiated. The user can overcome this limitation instantiating a parametric lambda:


.. code-block:: cpp
	:name: lambda-example2

	#include <hydra/FunctorWrapper.h>
	#include <hydra/Parameter.h>

	...

	auto raw_gaussian = [=] __host__ __device__ (unsigned int nparams, hydra::Parameters* params,
		unsigned int nargs, double* args) {

		double m2 = (x[0] - params[0] )*(x[0] - params[0] );
		double s2 = params[1]*params[1];
		
		return exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));

	};

	auto mean = hydra::Parameter::Create().Name("mean").Value(0.0).Limits(-1.0, 1.0).Error(0.01);

	auto sigma = hydra::Parameter::Create().Name("sigma").Value(1.0).Limits(0.01, 5.0).Error(0.01);

	auto wrapped_gaussian = hydra::wrap_lambda(raw_gaussian, mean, sigma);

	//set the parameters to different values 
	wrapped_gaussian.SetParameter(0, 1.0);
	wrapped_gaussian.SetParameter(1, 2.0);
	

The ``wrapped_gaussian`` of the previous example has the same functionality of the functor coded in the  :ref:`example <functor-example2>`.

Wrapped lambdas, parametric or not, also derives from ``hydra::BaseFunctor`` and provide the same functionality of the Hydra functors.

