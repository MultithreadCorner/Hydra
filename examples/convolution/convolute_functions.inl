/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * convolute_functions.inl
 *
 *  Created on: Nov 22, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CONVOLUTE_FUNCTIONS_INL_
#define CONVOLUTE_FUNCTIONS_INL_



#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>
#include <vector>

//hydra
#include <hydra/Convolution.h>
#include <hydra/functions/Gaussian.h>
#include <hydra/functions/UniformShape.h>

//command line
#include <tclap/CmdLine.h>

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

	double min=100;
	double max=200;

	auto nsamples = nentries;
	//===========================
	// kernels
	//---------------------------

	// gaussian
	auto mean   = hydra::Parameter::Create( "mean").Value(0.0).Error(0.0001);
	auto sigma  = hydra::Parameter::Create("sigma").Value(1.25).Error(0.0001);

	hydra::Gaussian<> gaussian_kernel(mean,  sigma);

	//===========================
	// signals
	//---------------------------

	auto A  = hydra::Parameter::Create("A").Value(130.0).Error(0.0001);
	auto B  = hydra::Parameter::Create("B").Value(175.0).Error(0.0001);

	hydra::UniformShape<> uniform_signal(A,B);


	//===========================
	// samples
	//---------------------------
	std::vector<double> conv_result(nsamples, 0.0);

	//uniform (x) gaussian kernel
	hydra::convolute(uniform_signal, gaussian_kernel, min, max,  conv_result );

	//------------------------
	//------------------------
#ifdef _ROOT_AVAILABLE_
	//fill histograms
	TH1D *hist     = new TH1D("signal", "signal", nsamples+1, min, max);

	for(size_t i=1;  i<nsamples+1; i++){

		hist->SetBinContent(i, conv_result[i] );
	}
#endif //_ROOT_AVAILABLE_






#ifdef _ROOT_AVAILABLE_
	TApplication *myapp=new TApplication("myapp",0,0);

	//----------------------------
	//draw histograms
	TCanvas* canvas = new TCanvas("canvas" ,"canvas", 500, 500);

	hist->SetStats(0);
	hist->SetLineColor(3);
	hist->SetLineWidth(2);
	hist->DrawNormalized("histl");

	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;

}


#endif /* CONVOLUTE_FUNCTIONS_INL_ */
