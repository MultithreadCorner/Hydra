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
This dramatically accelerates the calculations over large or multidimensional datasets. Hydra normalizes the pdfs on-the-fly using analytical
 or numerical integration algorithms provided by the framework and handles data using iterators. Hydra supports weigted and unweigted datasets. 

Hydra also provides an implementation of SPlot :cite:`splot`, a very popular technique for statistical unfolding of data distributions.


Defining PDFs
-------------

In Hydra, PDFs are represented by the ``hydra::Pdf<Functor, Integrator>`` class template and are defined binding a positive defined functor
 and a integrator. PDFs can be conveniently built using the template function
  
``hydra::make_pdf( pdf_object, integrator_object)``. 

Most of the functors provided in Hydra have an analytical integrator defined. To invoke it, one should use the class template  
``hydra::AnalyticalIntegral<Functor>``, otherwise an appropriated numerical integration algorithm needs be specified. 

The snippet below shows how bind a Gaussian to its analytical integrator and build a pdf object:

.. code-block:: cpp
	:name: pdf-gauss
		
	#include <hydra/Pdf.h>
	#include <hydra/Parameter.h>
	#include <hydra/functions/Gaussian.h>
    
	...

	//-----------------
    // some definitions
    double min   = -6.0;
    double max   =  6.0;

    //Parameters
    auto mean  = hydra::Parameter::Create("mean" ).Value(0.0).Error(0.0001).Limits(-1.0, 1.0);
    auto sigma = hydra::Parameter::Create("sigma").Value(1.0).Error(0.0001).Limits(0.01, 1.5);
    
    //Gaussian distribution 
    auto gauss = hydra::Gaussian<double>(mean, sigma);
    //Model
    auto model = hydra::make_pdf(gauss, hydra::AnalyticalIntegral< hydra::Gaussian<xvar> >(min, max) );

	...


It is also possible to represent models composed by the sum of two or more PDFs using the class templates  
``hydra::PDFSumExtendable<Pdf1, Pdf2,...>`` and  ``hydra::PDFSumNonExtendabl<Pdf1, Pdf2,...>`` .
Given N normalized pdfs :math:`F_i` , theses classes define objects representing the sum

.. math::

	F_t = \sum_i^N c_i \times F_i 

The coefficients :math:`c_i` can represent fractions or yields. If the number of coefficients is equal to
the number of PDFs, the coefficients are interpreted as yields and ``hydra::PDFSumExtendable<Pdf1, Pdf2,...>`` is used. If the number of coefficients is :math:`(N-1)`,
the class template ``hydra::PDFSumNonExtendabl<Pdf1, Pdf2,...>`` is used and the coefficients are interpreted as fractions defined in the interval [0,1].
The coefficient of the last term is calculated as :math:`c_N=1 -\sum_i^{(N-1)} c_i` .

``hydra::PDFSumExtendable<Pdf1, Pdf2,...>`` and  ``hydra::PDFSumNonExtendabl<Pdf1, Pdf2,...>`` objects can be conveniently created using the function template 
``hydra::add_pdfs(...)``.
 
The code snippet below shows how to implement a model with two components, a Gaussian and a Argus distribution,  
to build a extended model, which can be used to predict the corresponding yields:

.. code-block:: cpp
	:name: pdf-gauss-plus-argus
	
    #include <hydra/Parameter.h>
    #include <hydra/Pdf.h>
    #include <hydra/AddPdf.h> 
    #include <hydra/functions/Gaussian.h> 
    #include <hydra/functions/ArgusShape.h>
	...

	//-----------------
    // some definitions
    double min   =  5.20;
    double max   =  5.30;


    //===========================
    //fit model gaussian + argus

    //Gaussian
    hydra::Parameter  mean  = hydra::Parameter::Create().Name("Mean").Value( 5.28).Error(0.0001).Limits(5.25,5.29);
    hydra::Parameter  sigma = hydra::Parameter::Create().Name("Sigma").Value(0.0026).Error(0.0001).Limits(0.0024,0.0028);

    //gaussian function evaluating on the first argument
    auto Signal_PDF = hydra::make_pdf( hydra::Gaussian<_X>(mean, sigma),
            hydra::AnalyticalIntegral<hydra::Gaussian<_X>>(min, max));

    //-------------------------------------------
    //Argus
    //parameters
    auto  m0     = hydra::Parameter::Create().Name("M0").Value(5.291).Error(0.0001).Limits(5.28, 5.3);
    auto  slope  = hydra::Parameter::Create().Name("Slope").Value(-20.0).Error(0.0001).Limits(-30.0, -10.0);
    auto  power  = hydra::Parameter::Create().Name("Power").Value(0.5).Fixed();

    //gaussian function evaluating on the first argument
    auto Background_PDF = hydra::make_pdf( hydra::ArgusShape<_X>(m0, slope, power),
            hydra::AnalyticalIntegral<hydra::ArgusShape<_X>>(min, max));

    //------------------
    //yields
    hydra::Parameter N_Signal("N_Signal"        ,500, 100, 100 , nentries) ;
    hydra::Parameter N_Background("N_Background",2000, 100, 100 , nentries) ;

    //make model
    auto model = hydra::add_pdfs( {N_Signal, N_Background}, Signal_PDF, Background_PDF);
    model.SetExtended(1);

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



The Hydra classes representing PDFs are not dumb arithmetic beasts. 
These classes are lazy and implements a series of optimizations in order to forward to the thread collection only code that need effectively be evaluated.
In particular, functor normalization is cached in a such way that only new parameters settings will trigger the recalculation of integrals. 


Defining FCNs and invoking the ``ROOT::Minuit2`` interfaces
-----------------------------------------------------------

In general, a FCN is defined binding a PDF to the data the PDF is supposed to describe. 
Hydra implements classes and interfaces to allow the definition of FCNs suitable to perform maximum likelihood fits on unbinned and binned datasets.
The different use cases for Likelihood FCNs are covered by the specialization of the class template ``hydra::LogLikelihoodFCN<PDF, Iterator, Extensions...>``.

Objects representing  likelihood FCNs can be conveniently instantiated using the function template 
``hydra::make_likelihood_fcn(data_begin, data_end , PDF)``
 and ``hydra::make_likelihood_fcn(data_begin, data_end , weights_begin, PDF)``, 
 where ``data_begin``, ``data_end`` and ``weights_begin`` are iterators pointing to the dataset and the weights. 

.. code-block:: cpp
	
	#include <hydra/LogLikelihoodFCN.h>
    //Minuit2
    #include "Minuit2/FunctionMinimum.h"
    #include "Minuit2/MnUserParameterState.h"
    #include "Minuit2/MnPrint.h"
    #include "Minuit2/MnMigrad.h"
    #include "Minuit2/MnMinimize.h"
	...

	// get the fcn...
	auto fcn   = hydra::make_loglikehood_fcn(dataset.begin(), dataset.end(), model);
 	// and invoke Migrad minimizer from Minuit2
 	MnMigrad migrad(fcn, fcn.GetParameters().GetMnState(), MnStrategy(2));


sPlots
-------

The sPlot technique is used to unfold the contributions of different sources to the data sample in a given variable. The sPlot tool applies in the context of a Likelihood fit which needs to be performed on the data sample to determine the yields corresponding to the various sources. 

Hydra handles sPlots using the class template ``hydra::SPlot<Iterator, PDF1, PDF2,PDFs...>`` where ``Iterator`` is an iterator point to data
 ``PDF1``, ``PDF2`` and ``PDFs...``
 are the probability density functions describing the populations contributing to the dataset as modeled in a given
  variable referred as discriminating variable. 
 The other variables of interest, present in the dataset are referred as control variables and are 
statistically unfolded using the so called *sweights*. For each entry in the dataset, ``hydra::SPlot<Iterator, PDF1, PDF2,PDFs...>``
calculates a set of weights, each one corresponds to a data source described by the corresponding PDF.
It is not necessary to allocate memory to store the *sweights*. It is calculated on the fly when the user 
iterates over the ``hydra::Splot`` object. One can create the ``hydra::Splot`` object using the convenience 
functions  ``hydra::make_splot(PDF, data_range )``or ``hydra::make_splot(PDF, data_begin, data_end )``, where PDF is a 
``PDFSumExtendable<PDF1, PDF2, PDFs...>`` object.
It is responsability of the user to make sure that the passed ``PDF`` object properly optimized to describe the data.

.. code-block:: cpp

	#include <hydra/SPlot.h>

	...

    //splot
    //create splot
    auto sweigts = hydra::make_splot(fcn.GetPDF(), range );

    auto covar_matrix = sweigts.GetCovMatrix();


    std::cout << "Covariance matrix "
              << std::endl
              << covar_matrix
              << std::endl
              << std::endl;

    std::cout << std::endl
              << "sWeights:"
              << std::endl;

    for(size_t i = 0; i<10; i++)
        std::cout << "[" << i << "] :"
                  << sweigts[i]
                  << std::endl
                  << std::endl;


