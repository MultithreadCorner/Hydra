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
 * sample_distribution.inl
 *
 *  Created on: 20/07/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SAMPLE_DISTRIBUTION_INL_
#define SAMPLE_DISTRIBUTION_INL_

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
#include <hydra/FunctorArithmetic.h>
#include <hydra/Random.h>
#include <hydra/Copy.h>
#include <hydra/Tuple.h>
#include <hydra/Distance.h>
#include <hydra/multiarray.h>
#include <hydra/FunctionPtrBinderCPU.h>
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

double function(unsigned int n, double* p, double* x)
{
	return x[0]*x[1];
}

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


	//Gaussian 1
	double mean1   = -2.0;
	double sigma1  =  1.0;

	auto GAUSSIAN1D =  [=] __host__ __device__ (unsigned int n,double* x ){

		double g = 0.0;

			double m2 = (x[0] - mean1 )*(x[0] - mean1 );
			double s2 = sigma1*sigma1;
			g = exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));

		return g;
	};

	auto gaussian1D = hydra::wrap_lambda( GAUSSIAN1D );

	//==================

	auto GAUSSIAN1 =  [=] __host__ __device__ (unsigned int n,double* x ){

		double g = 1.0;

		for(size_t i=0; i<3; i++){

			double m2 = (x[i] - mean1 )*(x[i] - mean1 );
			double s2 = sigma1*sigma1;
			g *= exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));
		}

		return g;
	};

	auto gaussian1 = hydra::wrap_lambda( GAUSSIAN1 );

	//Gaussian 2
	double mean2   =  2.0;
	double sigma2  =  1.0;
	auto GAUSSIAN2 =  [=] __host__ __device__ (unsigned int n, double* x ){

		double g = 1.0;

		for(size_t i=0; i<3; i++){

			double m2 = (x[i] - mean2 )*(x[i] - mean2 );
			double s2 = sigma2*sigma2;
			g *= exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));
		}

		return g;
	};

	auto gaussian2 = hydra::wrap_lambda( GAUSSIAN2 );

	//sum of gaussians
	auto gaussians = gaussian1 + gaussian2;

	//---------

	//---------
	//generator
	hydra::Random<thrust::random::default_random_engine>
	Generator( std::chrono::system_clock::now().time_since_epoch().count() );

	hydra::mc_host_vector<double> vect(1000);

	auto middle = Generator.Sample(vect.begin(),  vect.end(), -6.0, 6.0, gaussians);

	std::array<double, 3>max{6.0, 6.0, 6.0};
	std::array<double, 3>min{-6.0, -6.0, -6.0};

	//------------------------
#ifdef _ROOT_AVAILABLE_

	TH3D hist_d("hist_d",   "3D Double Gaussian - Device",
			100, -6.0, 6.0,
			100, -6.0, 6.0,
			100, -6.0, 6.0 );

	TH3D hist_h("hist_h",   "3D Double Gaussian - Host",
			100, -6.0, 6.0,
			100, -6.0, 6.0,
			100, -6.0, 6.0	);

#endif //_ROOT_AVAILABLE_

	//some useful typedefs
	typedef hydra::tuple<double, double, double> row_t; //dataset entry
	typedef hydra::device::vector<row_t> table_d;       //vector of entries in device
	typedef hydra::host::vector<row_t> table_h;         //vector of entries in host
	typedef hydra::multivector<table_d> dataset_d;      //multivector
	typedef hydra::multivector<table_h> dataset_h;      //multivector

	//device
	{
		dataset_h data_h;
		{

			//we copy the accepted events and let
			//the data_d container go out scope
			//data_d will be destroyed, freeing
			//the allocated memory in the device ;)

			dataset_d data_d(nentries);

			auto middle = Generator.Sample(data_d.begin(),  data_d.end(), min, max, gaussians);

			size_t naccepted = hydra::distance(data_d.begin(), middle);

			data_h.resize( naccepted );
			hydra::copy(data_d.begin(), middle, data_h.begin());

		}

		std::cout <<std::endl;
		std::cout <<std::endl;
		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Sample > [" << i << "] :" << data_h[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_d.Fill( hydra::get<0>(value),
					hydra::get<1>(value),
					hydra::get<2>(value) );
#endif //_ROOT_AVAILABLE_

	}

	//host
	{

		dataset_h data_h(nentries);

		auto middle = Generator.Sample(data_h.begin(),  data_h.end(), min, max, gaussians);
		std::cout <<std::endl;
		std::cout <<std::endl;
		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Sample > [" << i << "] :" << data_h[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto itvalue = data_h.begin(); itvalue!= middle; itvalue++ ){
			auto value = *itvalue;
			hist_h.Fill( hydra::get<0>(value),
					hydra::get<1>(value),
					hydra::get<2>(value) );
		}
#endif //_ROOT_AVAILABLE_

	}


#ifdef _ROOT_AVAILABLE_
	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas_d("canvas_d" ,"Distributions - Device", 1000, 1000);
	hist_d.Draw("hist");

	//draw histograms
	TCanvas canvas_h("canvas_h" ,"Distributions - Host", 1000, 1000);
	hist_h.Draw("hist");

	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;



}


#endif /* SAMPLE_DISTRIBUTION_INL_ */
