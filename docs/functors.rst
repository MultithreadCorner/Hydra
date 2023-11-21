Functors and C++ lambdas.
===========================


The user's code is passed to Hydra's algorithms through functors and C++ lambdas.
Hydra then adds type information and functionality to functors and lambdas using CRTP.
Functors and lambdas are not attached to a specific back-end.
The signatures conventions adopted for functors and lambdas as well as the added functionality
will be discussed in the following lines.

Functors
--------

In C++, a functor, sometimes also referred as a function object, is any class or structure that overloads the function call operator ``òperator()(Args ...x)``.
In Hydra, all functors derives from the class template ``hydra::BaseFunctor<Functor, Signature, NParameters>``.
The template parameters are described below:

	* ``Functor`` : the type of the functor. 
	* ``Signature``: a type representing the signature of function call operaror. Ex: void(double, double).
	* ``NParameters``: the number of parameters the functor takes. 

The user needs to implement the method ``Evaluate(...)`` and Hydra will take care of implementing the function
call operator. To see how this works, it is convenient to give a look at the implementation
of the ``hydra::Gaussian`` functor:

    .. code-block:: cpp
    
    //Template parameters:
    //ArgType: the argument type. In this way, this also supports static named variables
    //Signature: whatever type it gets, the functor will return a double
    //Third parameter: 2 is the number of parameters a Gaussian takes (mean and width)
    template<typename ArgType, typename Signature=double(ArgType) >
    class Gaussian: public BaseFunctor<Gaussian<ArgType>, Signature, 2>
    {
        //import the parameters acessor _par
        using BaseFunctor<Gaussian<ArgType>, Signature, 2>::_par;

        public:
        //all members callable from host and device side
        //making sure that Gaussians always have a defined mean and sigma
        Gaussian()=delete;
    
        //constructors should also forward the parameters to BaseFunctor
        Gaussian(Parameter const& mean, Parameter const& sigma ):
            BaseFunctor<Gaussian<ArgType>, Signature, 2>({mean, sigma})
            {}
    
        __hydra_host__ __hydra_device__
        Gaussian(Gaussian<ArgType> const& other ):
            BaseFunctor<Gaussian<ArgType>, Signature, 2>(other)
            {}
    
        //operaror= should be always implemented 
        __hydra_host__ __hydra_device__
        Gaussian<ArgType>& operator=(Gaussian<ArgType> const& other )
        {
            if(this==&other) return  *this;
            BaseFunctor<Gaussian<ArgType>, Signature, 2>::operator=(other);
            return  *this;
        }
    
        //implement the evaluate method, where actual value of the functor for
        //its current parameters is calculated. 
        //CHECK_VALUE macro verifies the value and prints information in case of failure or NAN 
        __hydra_host__ __hydra_device__
        inline double Evaluate(ArgType x)  const
        {
            double m2 = ( x - _par[0])*(x - _par[0] );
            double s2 = _par[1]*_par[1];
            return CHECK_VALUE( ::exp(-m2/(2.0 * s2 )), "par[0]=%f, par[1]=%f", _par[0], _par[1]);
    
        }

};

Functors implemented in that fashion can deal with statically named variables and be optimized  
when fitting datasets. 

Hydra provides a growing set of native functors, which are available in  ``hydra/functions`` folder.
 
C++ Lambdas
-------------

Hydra supports C++ lambdas. Before to pass C++ lambdas to Hydra's algorithms,
users need to wrap it into a suitable Hydra object. This is done invoking the function template 
``hydra::wrap_lambda(...)``. Currently, lambdas with ``auto`` arguments are not supported.
Parametric lambdas, with or without named arguments are supported, though.  

	
	.. code-block:: cpp

	 auto multiply_by_two = hydra::wrap_lambda( 
	      [=] __hydra_dual__ ( double x){
		  
		       return 2.0*x;
		  } 
		) ;
	
	

Hydra can also handle "parametric lambdas". Parametric lambdas are wrapped
lambdas that can hold named parameters (``hydra::Parameters`` objecs). 
The signatures for parametric lambdas are:
    
    .. code-block:: cpp
    
    // mean 
    auto mean = hydra::Parameter::Create()
                        .Name("Mean_X")
                        .Value(0.0)
                        .Error(0.0001)
                        .Limits(-1.0, 1.0);

    // sigma 
    auto sigma = hydra::Parameter::Create()
                        .Name("Sigma_X")
                        .Value(2.0)
                        .Error(0.0001)
                        .Limits(0.1, 3.0);
    
    auto gaussian = hydra::wrap_lambda(
        [=] __hydra_dual__  (unsigned int npar, const hydra::Parameter* params, double x ) {

        double mean  = params[0].GetValue();
        double sigma = params[1].GetValue();
       
        double m2 = (X - mean ); m2 *= m2;
        double s2 = sigma*sigma;      

        return ::exp(-m2/(2.0 * s2 ))/( ::sqrt(2.0*s2*PI));

    }, mean, sigma);

