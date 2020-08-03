Parameter estimation
====================

The best Minuit description can be found on it's own user's manual :cite:`minuit` :

	Minuit is conceived as a tool to find the minimum value of a multi-parameter
	function, usually called “FCN”, and analyze the shape of this function around the minimum.
	The principal application is foreseen for statistical analysis, working on chi-square
	or log-likelihood functions, to compute the best-fit parameter values and uncertain-
	ties, including correlations between the parameters. It is especially suited to handle
	difficult problems, including those which may require guidance in order to find the
	correct solution. 

	-- Minuit User's Guide, Fred James and Matthias Winkler, June 16, 2004 -  CERN, Geneva.

Hydra implements an interface to Minuit2 that parallelizes the FCN calculation.
This dramatically accelerates the calculations over large data-sets. Hydra normalizes the pdfs on-the-fly using analytical or numerical integration algorithms provided by the framework and handles data using iterators. 

Hydra also provides an implementation of SPlot :cite:`splot`, a very popular technique for statistical unfolding of data distributions.


Defining PDFs
-------------

In Hydra, PDFs are represented by the ``hydra::Pdf<Functor, Integrator>`` class template and is defined binding a positive defined functor and a integrator. 
PDFs can be conveniently built using the template function 
``hydra::make_pdf( pdf, integrator)``. 
The snippet below shows how wrap a parametric lambda representing a Gaussian and bind it to a Gauss-Kronrod integrator, to build a pdf object:

.. code-block:: cpp
	:name: pdf-gauss
		
	#include <hydra/device/System.h>
	#include <hydra/Lambda.h>
	#include <hydra/Pdf.h>
	#include <hydra/Parameter.h>
	#include <hydra/GaussKronrodQuadrature.h>

	...

	std::string Mean("Mean"); 	// mean of gaussian
	std::string Sigma("Sigma"); // sigma of gaussian

	hydra::Parameter  mean_p  = hydra::Parameter::Create()
		.Name(Mean)
		.Value(0.5)
		.Error(0.0001)
		.Limits(-1.0, 1.0);

	hydra::Parameter  sigma_p = hydra::Parameter::Create()
		.Name(Sigma)
		.Value(0.5)
		.Error(0.0001)
		.Limits(0.01, 1.5);

	//wrap a parametric lambda 
	auto gaussian = hydra::wrap_lambda( [=] __host__ __device__ (unsigned int npar,
		const hydra::Parameter* params,  unsigned int narg, double* x ){

		double m2 = (x[0] -  params[0])*(x[0] - params[0] );
		double s2 = params[1]*params[1];
		
		return exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));
	}, mean_p, sigma_p);


	double min   = -5.0;  double max   =  5.0;

	//numerical integral to normalize the pdf
	hydra::GaussKronrodQuadrature<61,100, hydra::device::sys_t> GKQ61(min,  max);

	//build the PDF
	auto PDF = hydra::make_pdf(gaussian, GKQ61 );

	...


It is also possible to represent models composed by the sum of two or more PDFs using the class templates  
``hydra::PDFSumExtendable<Pdf1, Pdf2,...>`` and  ``hydra::PDFSumNonExtendabl<Pdf1, Pdf2,...>`` .
Given N normalized pdfs :math:`F_i` , theses classes define objects representing the sum

.. math::

	F_t = \sum_i^N c_i \times F_i 

The coefficients :math:`c_i` can represent fractions or yields. If the number of coefficients is equal to
the number of PDFs, the coefficients are interpreted as yields and ``hydra::PDFSumExtendable<Pdf1, Pdf2,...>`` is used. If the number of coefficients is :math:`(N-1)`,
the class template ``hydra::PDFSumNonExtendabl<Pdf1, Pdf2,...>`` is used and the coefficients are interpreted as fractions defined in the interval [0,1]. The coefficient of the last term is calculated as :math:`c_N=1 -\sum_i^{(N-1)} c_i` .

``hydra::PDFSumExtendable<Pdf1, Pdf2,...>`` and  ``hydra::PDFSumNonExtendabl<Pdf1, Pdf2,...>`` objects can be conveniently created using the function template 
``hydra::add_pdfs(...)``. 
The code snippet below continues the :ref:`example <pdf-gauss>` and defines a new PDF representing an exponential distribution and add it to the previous Gaussian PDF 
to build a extended model, which can be used to predict the yields:

.. code-block:: cpp
	:name: pdf-exponential

	...

	//tau of the exponential
	std::string  Tau("Tau");
	hydra::Parameter  tau_p  = hydra::Parameter::Create()
		.Name(Tau)
		.Value(1.0)
		.Error(0.0001)
		.Limits(-2.0, 2.0);

	//wrap a parametric lambda
	auto exponential = hydra::wrap_lambda( [=] __host__ __device__ (unsigned int npar,
	 	const hydra::Parameter* params,unsigned int narg, double* x ){
		
		double tau = params[0];
		return exp( -(x[0]-min)*tau);

	}, tau_p );

	// build the PDF
	auto PDF = hydra::make_pdf( exponential, GKQ61 );

	//yields
	std::string NG("N_Gauss");
	std::string NE("N_Exp");
	hydra::Parameter NG_p(NG , 1e4, 100.0, 1000 , 2e4) ;
	hydra::Parameter NE_p(NE , 1e4, 100.0, 1000 , 2e4) ;

	//add the pdfs
	auto model = hydra::add_pdfs({NG_p, NE_p}, gaussian, exponential );

	...


The user can get a reference to one of the component PDFs using the method ``PDF( hydra::placeholder )``. 
This is useful, for example, to change the state of a component PDF "in place". Same operation can 
be performed for coeficients using the method ``Coefficient( unsigned int )`` : 

.. code-block:: cpp
	
	#include<hydra/Placeholders.h>

	using namespace hydra::placeholders; 
	
	...

	//change the mean of the Gaussian to 2.0
	model.PDF( _0 ).SetParameter(0, 2.0);

	//set Gaussian coeficient  to 1.5e4
	model.Coefficient(0).SetValue(1.5e4);



The Hydra classes representing PDFs are not dumb arithmetic beasts. These classes are lazy and implements a series of optimizations in order to forward to the thread collection only code that need effectively be evaluated. In particular, functor normalization is cached in a such way that only new parameters settings will trigger the calculation of integrals. 


Defining FCNs and invoking the ``ROOT::Minuit2`` interfaces
-----------------------------------------------------------

In general, a FCN is defined binding a PDF to the data the PDF is supposed to describe. 
Hydra implements classes and interfaces to allow the definition of FCNs suitable to perform maximum likelihood fits on unbinned and binned datasets.
The different use cases for Likelihood FCNs are covered by the specialization of the class template ``hydra::LogLikelihoodFCN<PDF, Iterator, Extensions...>``.

Objects representing  likelihood FCNs can be conveniently instantiated using the function template ``hydra::make_likelihood_fcn(data_begin, data_end , PDF)`` and ``hydra::make_likelihood_fcn(data_begin, data_end , weights_begin, PDF)``, where ``data_begin``, ``data_end`` and ``weights_begin`` are iterators pointing to the dataset and the weights or bin-contents. 

.. code-block:: cpp
	
	#include <hydra/LogLikelihoodFCN.h>

	...

	// get the fcn...
	auto fcn   = hydra::make_loglikehood_fcn(dataset.begin(), dataset.end(), model);
 	// and invoke Migrad minimizer from Minuit2
 	MnMigrad migrad(fcn, fcn.GetParameters().GetMnState(), MnStrategy(2));


sPlots
-------

The sPlot technique is used to unfold the contributions of different sources to the data sample in a given variable. The sPlot tool applies in the context of a Likelihood fit which needs to be performed on the data sample to determine the yields corresponding to the various sources. 

Hydra handles sPlots using the class ``hydra::SPlot<PDF1, PDF2,PDFs...>`` where ``PDF1``, ``PDF2`` and ``PDFs...`` are the probability density functions describing the populations contributing to the dataset as modeled in a given variable referred as discriminating variable. The other variables of interest, present in the dataset are referred as control variables and are statistically unfolded using the so called *sweights*. For each entry in the dataset, ``hydra::SPlot<PDF1, PDF2,PDFs...>`` calculates a set of weights, where each one corresponds to a data source described by the corresponding PDF. It is responsibility of the user to allocate memory to store the *sweights*.

The weights are calculated invoking the method ``hydra::SPlot::Generate``, which returns the covariant matrix among the yields in the data sample.  

.. code-block:: cpp

	#include <hydra/SPlot.h>

	...

	//splot 2 components (gaussian + exponential )
	//hold weights
	hydra::multiarray<2, double, hydra::device::sys_t> sweigts(dataset.size());

	//create splot
	auto splot  = hydra::make_splot( fcn.GetPDF() );

	auto covarm = splot.Generate( dataset.begin(), dataset.end(), sweigts.begin());

