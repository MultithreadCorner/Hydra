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
#include <hydra/AddPdf.h>
#include <hydra/Copy.h>
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

    //fit
	auto GAUSSIAN =  [=] __host__ __device__
			(unsigned int npar, hydra::Parameter* params,unsigned int narg,double* x )
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

    //numerical integral
    hydra::GaussKronrodQuadrature<61,100, hydra::device::sys_t> GKQ61_d(min,  max);

	//device
	//------------------------

#ifdef _ROOT_AVAILABLE_

	TH1D hist_gaussian_d("gaussian_d", "Gaussian",    100, -6.0, 6.0);

#endif //_ROOT_AVAILABLE_

	{
		//1D device buffer
		hydra::device::vector<double>  data_d(nentries);
		hydra::host::vector<double>    data_h(nentries);

		//-------------------------------------------------------
		//gaussian
		Generator.Gauss(0.0, 1.0, data_d.begin(), data_d.end());
		hydra::copy(data_d.begin(), data_d.end(), data_h.begin());

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Gauss > [" << i << "] :" << data_d[i] << std::endl;

		//-------------------------------------------------------
		//fit


#ifdef _ROOT_AVAILABLE_
		for(auto value : data_d)
			hist_gaussian_d.Fill( value);
#endif //_ROOT_AVAILABLE_

	}



	//host
	//------------------------
#ifdef _ROOT_AVAILABLE_

	TH1D hist_gaussian_h("gaussian_h", "Gaussian",    100, -6.0, 6.0);

#endif //_ROOT_AVAILABLE_

	{
		//1D device buffer
		hydra::host::vector<double>    data_h(nentries);

		//-------------------------------------------------------
		//gaussian
		Generator.Gauss(0.0, 1.0, data_h.begin(), data_h.end());

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Gauss > [" << i << "] :" << data_h[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_gaussian_h.Fill( value);
#endif //_ROOT_AVAILABLE_

	}



#ifdef _ROOT_AVAILABLE_
	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas_d("canvas_d" ,"Distributions - Device", 1000, 1000);
	hist_gaussian_d.Draw("hist");


	//draw histograms
	TCanvas canvas_h("canvas_h" ,"Distributions - Host", 1000, 1000);
	hist_gaussian_h.Draw("hist");

	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;



}




#endif /* BASIC_FIT_INL_ */
