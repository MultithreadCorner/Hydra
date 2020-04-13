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
 * sample_distribution.inl
 *
 *  Created on: 20/07/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SAMPLE_DISTRIBUTION_INL_
#define SAMPLE_DISTRIBUTION_INL_

/**
 * \example sample_distribution.inl
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
#include <hydra/Lambda.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/Random.h>
#include <hydra/Algorithm.h>
#include <hydra/Tuple.h>
#include <hydra/Distance.h>
#include <hydra/multiarray.h>
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


	//Gaussian 1
	double mean    =  0.0;
	double sigmax  =  1.0;
	double sigmay  =  2.0;
	double sigmaz  =  3.0;

	auto Gauss3D =  hydra::wrap_lambda(
			[ mean, sigmax, sigmay, sigmaz] __hydra_dual__ (double x, double y, double z ) {

		    double g = 0.0;

			double mx_sq = (x - mean); mx_sq *=mx_sq;
			double sx_sq = sigmax*sigmax;
			double x_sq  = - 0.5*mx_sq/sx_sq;

			double my_sq = (y - mean); my_sq *=my_sq;
			double sy_sq = sigmay*sigmay;
			double y_sq  = - 0.5*my_sq/sy_sq;

			double mz_sq = (z - mean); mz_sq *=mz_sq;
			double sz_sq = sigmaz*sigmaz;
			double z_sq  = - 0.5*mz_sq/sz_sq;

			g = exp(x_sq + y_sq + z_sq);

		return g;
	});

	//---------


	std::array<double, 3> max{6.0, 6.0, 6.0};
	std::array<double, 3> min{-6.0, -6.0, -6.0};

	//------------------------
#ifdef _ROOT_AVAILABLE_

	TH3D histogram("histogram", "3D Gaussian",
		/*x */ 100,-6.0, 6.0, /*y */ 100,-6.0, 6.0, /*z */ 100,-6.0, 6.0 );

#endif //_ROOT_AVAILABLE_




	//device
	{

		hydra::multiarray<double,3,  hydra::device::sys_t> buffer(nentries);

		auto range = hydra::sample( hydra::device::sys, buffer.begin(),  buffer.end(), min, max, Gauss3D);

		auto Hist = hydra::make_dense_histogram<double,3>( hydra::device::sys,
						{100,100,100}, {-6.0, -6.0, -6.0}, {6.0, 6.0, 6.0},
						range);

#ifdef _ROOT_AVAILABLE_
		for(size_t i=0; i< 100; i++)
		{
			for(size_t j=0; j< 100; j++)
			{
				for(size_t k=0; k< 100; k++)
				{
					histogram.SetBinContent(i+1, j+1, k+1, Hist.GetBinContent({i,j,k}) );
				}
			}
		}

#endif //_ROOT_AVAILABLE_

	}



#ifdef _ROOT_AVAILABLE_
	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas("canvas" ,"", 1000, 1000);
	histogram.Draw("iso");
	histogram.SetFillColor(9);


	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;



}


#endif /* SAMPLE_DISTRIBUTION_INL_ */
