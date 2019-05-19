/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
 *
 *   This file is part of Hydra Data Analysis Framework.
 *
 *   Hydra is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   Hydra is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with Hydra.  If not, see <http://www.gnu.org/licenses/>.
 *
 *---------------------------------------------------------------------------*/

/*
 * fractional_logLL_fit.inl
 *
 *  Created on: 08/10/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FRACTIONAL_LOGLL_FIT_INL_
#define FRACTIONAL_LOGLL_FIT_INL_

/**
 * \example fractional_logLL_fit.inl
 *
 * This example show how to perform an unbinned likelihood
 * fit. The model has three components, two Gaussians and one Exponential,
 * \f$ model(x) = f_1*Gaussian_1(x) + f_2*Gaussian_2(x) + (1.0 - f_1 - f_2)*Exponential()\f$
 * The example first generating a dataset sampling the model in parallel and
 * then fit the parameters and fractions.
 */

#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>
#include <random>
#include <algorithm>

//command line
#include <tclap/CmdLine.h>

//this lib
#include <hydra/device/System.h>
#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Random.h>
#include <hydra/LogLikelihoodFCN.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/AddPdf.h>
#include <hydra/Filter.h>
#include <hydra/DenseHistogram.h>
#include <hydra/functions/Gaussian.h>
#include <hydra/functions/Exponential.h>

//Minuit2
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinimize.h"
#include "Minuit2/MnMinos.h"
#include "Minuit2/MnContours.h"
#include "Minuit2/CombinedMinimizer.h"
#include "Minuit2/MnPlot.h"
#include "Minuit2/MinosError.h"
#include "Minuit2/ContoursError.h"
#include "Minuit2/VariableMetricMinimizer.h"

/*-------------------------------------
 * Include classes from ROOT to fill
 * and draw histograms and plots.
 *-------------------------------------
 */
#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <TH1D.h>
#include <TApplication.h>
#include <TCanvas.h>

#endif //_ROOT_AVAILABLE_


using namespace ROOT::Minuit2;

int main(int argv, char** argc)
{
	size_t nentries = 0;

	try {

		TCLAP::CmdLine cmd("Command line arguments for ", '=');

		TCLAP::ValueArg<size_t> EArg("n", "number-of-events","Number of events", true, 10e6, "size_t");
		cmd.add(EArg);

		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		nentries = EArg.getValue();

	}
	catch (TCLAP::ArgException &e)  {
		std::cerr << "error: " << e.error() << " for arg " << e.argId()
														<< std::endl;
	}

	//-----------------
    // some definitions
    double min   =  0.0;
    double max   =  10.0;

	//generator
	hydra::Random<> Generator( std::chrono::system_clock::now().time_since_epoch().count() );

	//===========================
    //fit model
	hydra::Parameter  mean1_p  = hydra::Parameter::Create().Name("Mean_1").Value( 2.5) .Error(0.0001).Limits(0.0, 10.0);
	hydra::Parameter  sigma1_p = hydra::Parameter::Create().Name("Sigma_1").Value(0.5).Error(0.0001).Limits(0.01, 1.5);

	//gaussian function evaluating on the first argument
	hydra::Gaussian<0> gaussian1(mean1_p, sigma1_p);
	auto Gauss1_PDF = hydra::make_pdf(gaussian1, hydra::AnalyticalIntegral<hydra::Gaussian<0>>(min, max));

    //-------------------------------------------

    //gaussian 2
    hydra::Parameter  mean2_p  = hydra::Parameter::Create().Name("Mean_2").Value(5.0) .Error(0.0001).Limits(0.0, 10.0);
    hydra::Parameter  sigma2_p = hydra::Parameter::Create().Name("Sigma_2").Value(0.5).Error(0.0001).Limits(0.01, 1.5);

    //gaussian function evaluating on the first argument
    hydra::Gaussian<0> gaussian2(mean2_p, sigma2_p);
    auto Gauss2_PDF = hydra::make_pdf(gaussian2, hydra::AnalyticalIntegral<hydra::Gaussian<0>>(min, max));

    //--------------------------------------------

    //exponential
    //parameters
    hydra::Parameter  tau_p  = hydra::Parameter::Create().Name("Tau").Value(1.0) .Error(0.0001).Limits(-2.0, 2.0);

    //gaussian function evaluating on the first argument
    hydra::Exponential<0> exponential(tau_p);
    auto Exp_PDF = hydra::make_pdf(exponential, hydra::AnalyticalIntegral<hydra::Exponential<0>>(min, max));

    //------------------
    //yields
	hydra::Parameter F_Gauss_1_p("F_Gauss1" ,0.5, 0.001, 0.1 , 0.5) ;
	hydra::Parameter F_Gauss_2_p("F_Gauss2" ,0.5, 0.001, 0.1 , 0.5) ;

	//make model
	auto model = hydra::add_pdfs( std::array<hydra::Parameter,2>{F_Gauss_1_p, F_Gauss_2_p }, Gauss1_PDF, Gauss2_PDF, Exp_PDF);

	//===========================

#ifdef _ROOT_AVAILABLE_

	TH1D hist_gaussian_d("gaussian_d", "Gaussian",    100, min, max);
	TH1D hist_fitted_gaussian_d("fitted_gaussian_d", "Gaussian",    100, min, max);

#endif //_ROOT_AVAILABLE_

	//scope begin
	{

		//1D device buffer
		hydra::device::vector<double>  data_d(3*nentries);

		//-------------------------------------------------------
		// Generate data

		// gaussian1
		Generator.Gauss(mean1_p.GetValue()+0.5, sigma1_p.GetValue()+0.5, data_d.begin(), data_d.begin()+nentries);

		// gaussian1
		Generator.Gauss(mean2_p.GetValue()+0.5, sigma2_p.GetValue()+0.5, data_d.begin()+nentries, data_d.begin()+2*nentries);

		// exponential
		Generator.Exp(tau_p.GetValue()+0.5, data_d.begin() + 2*nentries,  data_d.end());

		std::cout<< std::endl<< "Generated data:"<< std::endl;
		for(size_t i=0; i<10; i++)
			std::cout << "[" << i << "] :" << data_d[i] << std::endl;


		//filtering
		auto filter = hydra::wrap_lambda(
			[=] __hydra_dual__ (unsigned int n, double* x){
				return (x[0] > min) && (x[0] < max );
		});

		auto range  = hydra::apply_filter(data_d,  filter);

		std::cout<< std::endl<< "Filtered data:"<< std::endl;
		for(size_t i=0; i<10; i++)
			std::cout << "[" << i << "] :" << range.begin()[i] << std::endl;


		//make model and fcn
		auto fcn   = hydra::make_loglikehood_fcn(model, range.begin(), range.end() );

		//-------------------------------------------------------
		//fit
		ROOT::Minuit2::MnPrint::SetLevel(3);
		hydra::Print::SetLevel(hydra::WARNING);
		//minimization strategy
		MnStrategy strategy(2);

		// create Migrad minimizer
		MnMigrad migrad_d(fcn, fcn.GetParameters().GetMnState() ,  strategy);

		std::cout<<fcn.GetParameters().GetMnState()<<std::endl;

		// ... Minimize and profile the time

		auto start_d = std::chrono::high_resolution_clock::now();
		FunctionMinimum minimum_d =  FunctionMinimum(migrad_d(std::numeric_limits<unsigned int>::max(), 5));
		auto end_d = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_d = end_d - start_d;

		// output
		std::cout<<"Minimum: "<< minimum_d << std::endl;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| [Fit] GPU Time (ms) ="<< elapsed_d.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;


		//--------------------------------------------
		hydra::DenseHistogram<double, 1, hydra::device::sys_t> Hist_Data(100, min, max);
		Hist_Data.Fill( range.begin(), range.end() );

#ifdef _ROOT_AVAILABLE_

		for(size_t i=0;  i<100; i++)
			 hist_gaussian_d.SetBinContent(i+1, Hist_Data.GetBinContent(i));

		//draw fitted function
		hist_fitted_gaussian_d.Sumw2();
		for (size_t i=0 ; i<=100 ; i++) {
			double x = hist_fitted_gaussian_d.GetBinCenter(i);
	        hist_fitted_gaussian_d.SetBinContent(i, fcn.GetPDF()(x) );
		}
		hist_fitted_gaussian_d.Scale(hist_gaussian_d.Integral()/hist_fitted_gaussian_d.Integral() );
#endif //_ROOT_AVAILABLE_

	}//scope end





#ifdef _ROOT_AVAILABLE_
	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas_d("canvas_d" ,"Distributions - Device", 500, 500);
	hist_gaussian_d.Draw("hist");
	hist_fitted_gaussian_d.Draw("histsameC");
	hist_fitted_gaussian_d.SetLineColor(2);

	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;



}



#endif /* FRACTIONAL_LOGLL_FIT_INL_ */
