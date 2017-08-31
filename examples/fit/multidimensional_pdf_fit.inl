/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 Antonio Augusto Alves Junior
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
 * basic_fit.inl
 *
 *  Created on: 14/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef BASIC_FIT_INL_
#define BASIC_FIT_INL_


#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>

//command line
#include <tclap/CmdLine.h>

//this lib
#include <hydra/device/System.h>
#include <hydra/host/System.h>
#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Random.h>
#include <hydra/LogLikelihoodFCN2.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/Copy.h>
#include <hydra/Filter.h>
#include <hydra/GaussKronrodQuadrature.h>

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


	//generator
	hydra::Random<thrust::random::default_random_engine>
	Generator( std::chrono::system_clock::now().time_since_epoch().count() );

	//----------------------
    //fit function
	auto GAUSSIAN =  [=] __host__ __device__
			(unsigned int npar, const hydra::Parameter* params,unsigned int narg, double* x )
	{
		double m2 = (x[0] -  params[0])*(x[0] - params[0] );
		double s2 = params[1]*params[1];
		double g = exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));

		return g;
	};


	std::string Mean("Mean"); 	// mean of gaussian
	std::string Sigma("Sigma"); // sigma of gaussian
	hydra::Parameter  mean_p  = hydra::Parameter::Create().Name(Mean).Value(0.5) .Error(0.0001).Limits(-1.0, 1.0);
	hydra::Parameter  sigma_p = hydra::Parameter::Create().Name(Sigma).Value(0.5).Error(0.0001).Limits(0.01, 1.5);

    auto gaussian = hydra::wrap_lambda(GAUSSIAN, mean_p, sigma_p);

    //-----------------
    // some definitions
    double min   = -5.0;
    double max   =  5.0;
    double mean  = 0.0;
    double sigma = 1.0;

	//device
	//------------------------
#ifdef _ROOT_AVAILABLE_

	TH1D hist_gaussian_d("gaussian_d", "Gaussian",    100, min, max);
	TH1D hist_fitted_gaussian_d("fitted_gaussian_d", "Gaussian",    100, min, max);

#endif //_ROOT_AVAILABLE_
	{
		//1D device buffer
		hydra::device::vector<double>  data_d(nentries);
		hydra::host::vector<double>    data_h(nentries);

		//-------------------------------------------------------
		//gaussian
		Generator.Gauss(mean, sigma, data_d.begin(), data_d.end());

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Gauss > [" << i << "] :" << data_d[i] << std::endl;

		//numerical integral to normalize the pdf
		hydra::GaussKronrodQuadrature<61,100, hydra::device::sys_t> GKQ61_d(min,  max);

		//filtering
		auto FILTER = [=]__host__ __device__(unsigned int n, double* x){
			return (x[0] > min) && (x[0] < max );
		};

		auto filter = hydra::wrap_lambda(FILTER);
		auto range  = hydra::apply_filter(data_d,  filter);

		std::cout<< std::endl<< std::endl;
		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Gauss > [" << i << "] :" << range.first[i] << std::endl;


		//make model and fcn
		auto model = hydra::make_pdf(gaussian, GKQ61_d );
		auto fcn   = hydra::make_loglikehood_fcn(range.first, range.second, model);

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
		std::cout<<"minimum: "<<minimum_d<<std::endl;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| GPU Time (ms) ="<< elapsed_d.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

		//bring data to device
		hydra::copy( data_d.begin() , data_d.end(), data_h.begin() );

		//draw fitted function


#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_gaussian_d.Fill( value);

		//draw fitted function
		hist_fitted_gaussian_d.Sumw2();
		for (size_t i=0 ; i<=100 ; i++) {
			double x = hist_fitted_gaussian_d.GetBinCenter(i);
	        hist_fitted_gaussian_d.SetBinContent(i, fcn.GetPDF()(x) );
		}
		hist_fitted_gaussian_d.Scale(hist_gaussian_d.Integral()/hist_fitted_gaussian_d.Integral() );
#endif //_ROOT_AVAILABLE_

	}//device end



	//host
	//------------------------
#ifdef _ROOT_AVAILABLE_

	TH1D hist_gaussian_h("gaussian_h", "Gaussian",    100, min, max );
	TH1D hist_fitted_gaussian_h("fitted_gaussian_h", "Gaussian",    100, min, max);
#endif //_ROOT_AVAILABLE_

	{
		//1D device buffer
		hydra::host::vector<double>    data_h(nentries);

		//-------------------------------------------------------
		//gaussian
		Generator.Gauss(mean, sigma, data_h.begin(), data_h.end());

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Gauss > [" << i << "] :" << data_h[i] << std::endl;

		//numerical integral to normalize the pdf
		hydra::GaussKronrodQuadrature<61,100, hydra::host::sys_t> GKQ61_h(min,  max);

		//filtering
		auto FILTER = [=]__host__ __device__(unsigned int n, double* x){
			return (x[0] > min) && (x[0] < max );
		};

		auto filter = hydra::wrap_lambda(FILTER);
		auto range  = hydra::apply_filter(data_h,  filter);

		std::cout<< std::endl<< std::endl;
		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Gauss > [" << i << "] :" << range.first[i] << std::endl;


		//make model and fcn
		auto model = hydra::make_pdf(gaussian, GKQ61_h );
		auto fcn   = hydra::make_loglikehood_fcn(range.first, range.second, model);

		//-------------------------------------------------------
		//fit
		ROOT::Minuit2::MnPrint::SetLevel(3);
		//minimization strategy
		MnStrategy strategy(2);

		// create Migrad minimizer
		MnMigrad migrad_h(fcn, fcn.GetParameters().GetMnState() ,  strategy);

		std::cout << fcn.GetParameters().GetMnState() << std::endl;

		// ... Minimize and profile the time
		auto start_h = std::chrono::high_resolution_clock::now();
		FunctionMinimum minimum_h =  FunctionMinimum(migrad_h(std::numeric_limits<unsigned int>::max(), 5));
		auto end_h = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_h = end_h - start_h;

		// output
		std::cout<<"minimum: "<< minimum_h <<std::endl;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| GPU Time (ms) ="<< elapsed_h.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;


		//draw fitted function
#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_gaussian_h.Fill( value);

		//draw fitted function
		hist_fitted_gaussian_h.Sumw2();
		for (size_t i=0 ; i<=100 ; i++) {
			double x = hist_fitted_gaussian_h.GetBinCenter(i);
			hist_fitted_gaussian_h.SetBinContent(i, fcn.GetPDF()(x) );
		}
		hist_fitted_gaussian_h.Scale(hist_gaussian_h.Integral()/hist_fitted_gaussian_h.Integral() );
#endif //_ROOT_AVAILABLE_

	}



#ifdef _ROOT_AVAILABLE_
	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas_d("canvas_d" ,"Distributions - Device", 500, 500);
	hist_gaussian_d.Draw("hist");
	hist_fitted_gaussian_d.Draw("histsameC");
	hist_fitted_gaussian_d.SetLineColor(2);

	//draw histograms
	TCanvas canvas_h("canvas_h" ,"Distributions - Host", 500, 500);
	hist_gaussian_h.Draw("hist");
	hist_fitted_gaussian_h.Draw("histsameC");
	hist_fitted_gaussian_h.SetLineColor(2);

	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;



}




#endif /* BASIC_FIT_INL_ */
