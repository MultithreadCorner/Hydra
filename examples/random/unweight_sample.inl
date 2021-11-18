/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2021 Antonio Augusto Alves Junior
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
 * unweight_sample.inl
 *
 *  Created on: 21/10/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef UNWEIGHT_SAMPLE_INL_
#define UNWEIGHT_SAMPLE_INL_


/**
 * \example unweight_distribution.inl
 * \brief This example shows how to produce an un-weighted
 * sample, with a 2D-Gaussian shape, from a flat 2D sample
 * using a gaussian functor. *
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
#include <hydra/Random.h>
#include <hydra/Algorithm.h>
#include <hydra/Tuple.h>
#include <hydra/multiarray.h>
#include <hydra/functions/Gaussian.h>
#include <hydra/functions/Exponential.h>
#include <hydra/functions/UniformShape.h>
#include <hydra/DenseHistogram.h>
/*-------------------------------------
 * Include classes from ROOT to fill
 * and draw histograms and plots.
 *-------------------------------------
 */
#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <TH3D.h>
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

	unsigned nrbins = 100;
	double min     =  5.0;
	double max     = 20.0;
    double mean    =  (max-min)*0.5+min;
	double sigma   =  1.0;

	//------------------------
#ifdef _ROOT_AVAILABLE_

	TH1D histo_flat("flat", "flat sample", nrbins, min, max);

	TH1D histo_gauss("gauss", "gaussian sample", nrbins, min, max);

	TH1D histo_exp("exp", "exponential sample", nrbins, min, max);

#endif //_ROOT_AVAILABLE_

	//Uniform
	auto A = hydra::Parameter::Create().Name("A").Value(min);
	auto B = hydra::Parameter::Create().Name("B").Value(max);
	auto uniform   = hydra::UniformShape<double>(A,B);

	//Gaussian distribution
	//Parameters
	auto Mean  = hydra::Parameter::Create("mean" ).Value(mean);
	auto Sigma = hydra::Parameter::Create("sigma").Value(sigma);
	auto gauss = hydra::Gaussian<double>(Mean, Sigma);

	//Exponential distribution
	auto tau  = hydra::Parameter::Create("tau" ).Value(1.0);
	auto exp  = hydra::Exponential<double>(tau);

	//device
	{

		hydra::device::vector<double> flat(nentries);

		auto start = std::chrono::high_resolution_clock::now();

		hydra::fill_random(flat, uniform);

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "----------------- Device ----------------"<< std::endl;
		std::cout << "| Uniform filling                        "<< std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;

		auto Hist_Flat = hydra::make_dense_histogram<double>( hydra::device::sys, nrbins, min, max, flat);

		//------------------

		start = std::chrono::high_resolution_clock::now();

		auto range_gauss = hydra::unweight(flat, gauss, gauss(mean), 0x8ec74d321e6b5a27, 2*nentries);

		end = std::chrono::high_resolution_clock::now();

		elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "----------------- Device ----------------"<< std::endl;
		std::cout << "| Gauss (unweight)                         "<< std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;

		auto Hist_Gauss = hydra::make_dense_histogram<double>( hydra::device::sys, nrbins, min, max, range_gauss);


		//------------------

		start = std::chrono::high_resolution_clock::now();

		auto range_exp = hydra::unweight(flat, exp, exp(0.0), 0x8ec74d321e6b5a27, 3*nentries);

		end = std::chrono::high_resolution_clock::now();

		elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "----------------- Device ----------------"<< std::endl;
		std::cout << "| Exp (unweight)                           "<< std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;

		auto Hist_Exp = hydra::make_dense_histogram<double>( hydra::device::sys, nrbins, min, max, range_exp);



#ifdef _ROOT_AVAILABLE_
		for(size_t i=0; i< nrbins; ++i){
			histo_flat.SetBinContent(i+1, Hist_Flat.GetBinContent(i) );
			histo_gauss.SetBinContent(i+1, Hist_Gauss.GetBinContent(i) );
			histo_exp.SetBinContent(i+1, Hist_Exp.GetBinContent(i) );
		}
#endif //_ROOT_AVAILABLE_



	}



#ifdef _ROOT_AVAILABLE_
	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas1("flat" ,"", 1000, 1000);
	histo_flat.Draw("hist");

	TCanvas canvas2("gauss" ,"", 1000, 1000);
	histo_gauss.Draw("hist");

	TCanvas canvas3("exp" ,"", 1000, 1000);
	histo_exp.Draw("hist");

	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;



}



#endif /* UNWEIGHT_SAMPLE_INL_ */
