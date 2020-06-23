/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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

/**
 * \example basic_fit.inl
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
#include <hydra/LogLikelihoodFCN.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/functions/Gaussian.h>
#include <hydra/DenseHistogram.h>
#include <hydra/multivector.h>
#include <hydra/Zip.h>

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
using namespace hydra::arguments;


declarg(xvar, double)

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
		std::cerr << " error: "  << e.error()
				  << " for arg " << e.argId()
				  << std::endl;
	}

	//-----------------
	// some definitions
	double min   = -6.0;
	double max   =  6.0;

	//Parameters for X direction
	auto xmean  = hydra::Parameter::Create("X-mean" ).Value(0.0).Error(0.0001).Limits(-1.0, 1.0);
	auto xsigma = hydra::Parameter::Create("X-sigma").Value(1.0).Error(0.0001).Limits(0.01, 1.5);
    //Gaussian distribution for X direction
	auto xgauss = hydra::Gaussian<xvar>(xmean, xsigma);
	//Model for X direction
	auto xmodel = hydra::make_pdf(xgauss, hydra::AnalyticalIntegral< hydra::Gaussian<xvar> >(min, max) );


	//------------------------
#ifdef _ROOT_AVAILABLE_

	TH1D hist_xvar("hist_xvar", "X-axis", 100, min, max);

	TH1D hist_fit_xvar("hist_fit_xvar", "X-axis", 100, min, max);

#endif //_ROOT_AVAILABLE_

	//begin raii scope
	{


		//1D device buffer
		hydra::device::vector<xvar> dataset(nentries);
		//-------------------------------------------------------

		//gaussian range

		hydra::copy(hydra::random_gauss_range(xmean.GetValue(), xsigma.GetValue(), 159753,nentries ), dataset);

		std::cout<< std::endl<< "Generated data:"<< std::endl;
		for(size_t i=0; i<10; i++)
			std::cout << "[" << i << "] :" << dataset[i] << std::endl;

		auto xfcn   = hydra::make_loglikehood_fcn(xmodel, dataset);

		//-------------------------------------------------------
		//fit

		ROOT::Minuit2::MnPrint::SetLevel(3);
		hydra::Print::SetLevel(hydra::WARNING);

		//minimization strategy
		MnStrategy strategy(2);

		//create Migrad minimizer
		MnMigrad xmigrad(xfcn, xfcn.GetParameters().GetMnState() ,  strategy);

		//print parameters before fitting
		std::cout << xfcn.GetParameters().GetMnState() << std::endl;

		//Minimize and profile the time
		auto start = std::chrono::high_resolution_clock::now();

		FunctionMinimum xminimum = FunctionMinimum( xmigrad(std::numeric_limits<unsigned int>::max(), 5));

		auto stop  = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = stop - start;

		//print minuit result
		std::cout << " minimum: " << xminimum << std::endl;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| Time (ms) ="<< elapsed.count()    <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

		hydra::DenseHistogram<double, 1, hydra::device::sys_t>Hist_Data(100, min, max );

		Hist_Data.Fill( dataset );


#ifdef _ROOT_AVAILABLE_
		//draw data
		for(size_t i=0;  i<100; i++){
			hist_xvar.SetBinContent(i+1, Hist_Data.GetBinContent(i)  );
		}

		//draw fitted function
		hist_fit_xvar.Sumw2();
		for (size_t i=0 ; i<=100 ; i++) {
			double x = hist_fit_xvar.GetBinCenter(i);
	         hist_fit_xvar.SetBinContent(i, xfcn.GetPDF()(x) );
		}

		hist_fit_xvar.Scale(hist_xvar.Integral()/hist_fit_xvar.Integral() );
#endif //_ROOT_AVAILABLE_

	}//end raii scope


#ifdef _ROOT_AVAILABLE_
	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas("canvas_d" ,"Distributions - Device", 500, 500);
	hist_xvar.Draw("hist");
	hist_fit_xvar.Draw("histsameC");
	hist_fit_xvar.SetLineColor(2);

	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;
}
#endif /* BASIC_FIT_INL_ */
