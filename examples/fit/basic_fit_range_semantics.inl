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
 * basic_fit_range_semantics.inl
 *
 *  Created on: 01/07/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef BASIC_FIT_RANGE_SEMANTICS_INL_
#define BASIC_FIT_RANGE_SEMANTICS_INL_


/**
 * \example basic_fit_range_semantics.inl
 *
 * This example shows how to generate a normal distributed dataset
 * and fit a hydra::Gaussian distribution.
 *
 */

#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>

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
#include <hydra/Filter.h>
#include <hydra/functions/Gaussian.h>
#include <hydra/DenseHistogram.h>
//Minuit2
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinimize.h"

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
	double min   = -5.0;
	double max   =  5.0;
	double mean  =  0.0;
	double sigma =  1.0;

	//generator
	hydra::Random<> Generator( std::chrono::system_clock::now().time_since_epoch().count() );

	//parameters
	hydra::Parameter  mean_p  = hydra::Parameter::Create().Name("Mean").Value(0.5).Error(0.0001).Limits(-1.0, 1.0);
	hydra::Parameter  sigma_p = hydra::Parameter::Create().Name("Sigma").Value(0.5).Error(0.0001).Limits(0.01, 1.5);


	//gaussian function evaluating on argument zero
	hydra::Gaussian<> gaussian(mean_p,sigma_p);

	//make model (pdf with analytical integral)
	auto model = hydra::make_pdf(gaussian, hydra::AnalyticalIntegral<hydra::Gaussian<>>(min, max) );


	//------------------------
#ifdef _ROOT_AVAILABLE_

	TH1D hist_gaussian_d("gaussian_d", "Gaussian",    100, min, max);
	TH1D hist_fitted_gaussian_d("fitted_gaussian_d", "Gaussian",    100, min, max);

#endif //_ROOT_AVAILABLE_

	//begin raii scope
	{

		//1D device buffer
		hydra::device::vector<double>  data_d(nentries);

		//-------------------------------------------------------
		//gaussian
		Generator.Gauss(mean, sigma, data_d.begin(), data_d.end());

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
			std::cout << "[" << i << "] :" << range[i] << std::endl;

		auto fcn   = hydra::make_loglikehood_fcn(model, range);

		//-------------------------------------------------------
		//fit

		ROOT::Minuit2::MnPrint::SetLevel(3);
		hydra::Print::SetLevel(hydra::WARNING);

		//minimization strategy
		MnStrategy strategy(2);

		//create Migrad minimizer
		MnMigrad migrad_d(fcn, fcn.GetParameters().GetMnState() ,  strategy);

		//print parameters before fitting
		std::cout<<fcn.GetParameters().GetMnState()<<std::endl;

		//Minimize and profile the time
		auto start_d = std::chrono::high_resolution_clock::now();

		FunctionMinimum minimum_d =  FunctionMinimum(migrad_d(std::numeric_limits<unsigned int>::max(), 5));

		auto end_d = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed_d = end_d - start_d;

		//print minuit result
		std::cout<<"minimum: "<<minimum_d<<std::endl;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| GPU Time (ms) ="<< elapsed_d.count()    <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

		hydra::DenseHistogram<double,1,  hydra::device::sys_t> Hist_Data(100, min, max);
		Hist_Data.Fill( range );


#ifdef _ROOT_AVAILABLE_
		//draw data
		for(size_t i=0;  i<100; i++){
			hist_gaussian_d.SetBinContent(i+1, Hist_Data.GetBinContent(i)  );
		}

		//draw fitted function
		hist_fitted_gaussian_d.Sumw2();
		for (size_t i=0 ; i<=100 ; i++) {
			double x = hist_fitted_gaussian_d.GetBinCenter(i);
	        hist_fitted_gaussian_d.SetBinContent(i, fcn.GetPDF()(x) );
		}

		hist_fitted_gaussian_d.Scale(hist_gaussian_d.Integral()/hist_fitted_gaussian_d.Integral() );
#endif //_ROOT_AVAILABLE_

	}//end raii scope


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



#endif /* BASIC_FIT_RANGE_SEMANTICS_INL_ */
