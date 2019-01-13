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
 * basic_distributions.inl
 *
 *  Created on: Jul 18, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef BASIC_DISTRIBUTIONS_INL_
#define BASIC_DISTRIBUTIONS_INL_

/**
 * \example basic_distributions.inl
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
#include <hydra/host/System.h>
#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Random.h>
#include <hydra/Algorithm.h>

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


	//generator
	hydra::Random<>
	Generator( std::chrono::system_clock::now().time_since_epoch().count() );


	//device
	//------------------------
#ifdef _ROOT_AVAILABLE_

	TH1D hist_uniform_d("uniform_d",   "Uniform",     100, -6.0, 6.0);
	TH1D hist_gaussian_d("gaussian_d", "Gaussian",    100, -6.0, 6.0);
	TH1D hist_exp_d("exponential_d",   "Exponential", 100,  0.0, 5.0);
	TH1D hist_bw_d("breit_wigner_d",   "Breit-Wigner",100,  0.0, 5.0);

#endif //_ROOT_AVAILABLE_

	{
		//1D device buffer
		hydra::device::vector<double>  data_d(nentries);
		hydra::host::vector<double>    data_h(nentries);

		//-------------------------------------------------------
		//uniform
		Generator.Uniform(-5.0, 5.0, data_d.begin(), data_d.end());
		hydra::copy(data_d,  data_h);

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Uniform > [" << i << "] :" << data_d[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_uniform_d.Fill( value);
#endif //_ROOT_AVAILABLE_

		//-------------------------------------------------------
		//gaussian
		Generator.Gauss(0.0, 1.0, data_d.begin(), data_d.end());
		hydra::copy(data_d,  data_h);

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Gauss > [" << i << "] :" << data_d[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_d)
			hist_gaussian_d.Fill( value);
#endif //_ROOT_AVAILABLE_

		//-------------------------------------------------------
		//exponential
		Generator.Exp(1.0, data_d);
		hydra::copy(data_d,  data_h);

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Exp > [" << i << "] :" << data_d[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_exp_d.Fill( value);
#endif //_ROOT_AVAILABLE_

		//-------------------------------------------------------
		//breit-wigner
		Generator.BreitWigner(2.0, 0.2, data_d.begin(), data_d.end());
		hydra::copy(data_d,  data_h);

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::BreitWigner > [" << i << "] :" << data_d[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_bw_d.Fill( value);
#endif //_ROOT_AVAILABLE_
	}




#ifdef _ROOT_AVAILABLE_
	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas_d("canvas_d" ,"Distributions - Device", 1000, 1000);
	canvas_d.Divide(2,2);
	canvas_d.cd(1); hist_uniform_d.Draw("hist");
	canvas_d.cd(2); hist_gaussian_d.Draw("hist");
	canvas_d.cd(3); hist_exp_d.Draw("hist");
	canvas_d.cd(4); hist_bw_d.Draw("hist");



	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;



}



#endif /* BASIC_DISTRIBUTIONS_INL_ */
