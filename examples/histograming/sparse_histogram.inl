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
 * sparse_histograming.inl
 *
 *  Created on: 03/10/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPARSE_HISTOGRAMING_INL_
#define SPARSE_HISTOGRAMING_INL_

/**
 * \example dense_histogram.inl
 *
 * This example shows the basic usage of the hydra::DenseHistogram class.
 */

#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>

//command line
#include <tclap/CmdLine.h>

//this lib
#include <hydra/host/System.h>
#include <hydra/device/System.h>
#include <hydra/Function.h>
#include <hydra/Lambda.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/Random.h>
#include <hydra/Algorithm.h>
#include <hydra/Tuple.h>
#include <hydra/Distance.h>
#include <hydra/SparseHistogram.h>
#include <hydra/Range.h>
#include <hydra/multivector.h>

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

declarg(AxisX, double)
declarg(AxisY, double)
declarg(AxisZ, double)

using namespace hydra::arguments;

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


	//==================

	//Gaussian 1
	double mean1   =  1.0;
	double sigma1  =  1.0;

	auto Gaussian1 = hydra::wrap_lambda( [mean1, sigma1] __hydra_dual__ ( AxisX x, AxisY y, AxisZ z )
	{

		double g = 1.0;

		double X[3]{x,y,z};

		for(size_t i=0; i<3; i++)
		{

			double m2 = (X[i] - mean1 );
			m2=m2*m2;
			double s2 = sigma1*sigma1;
			g *= exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));
		}

		return g;
	});


	//Gaussian 2
	double mean2   =  3.0;
	double sigma2  =  1.0;
	auto Gaussian2 = hydra::wrap_lambda( [mean2, sigma2] __hydra_dual__ ( AxisX x, AxisY y, AxisZ z )
		{

			double g = 1.0;

			double X[3]{x,y,z};

			for(size_t i=0; i<3; i++)
			{

				double m2 = (X[i] - mean2 );
				m2=m2*m2;
				double s2 = sigma2*sigma2;
				g *= exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));
			}

			return g;
		});

	//sum of gaussians
	auto Gaussians = Gaussian1 + Gaussian2;

	//---------
	//---------
	//generator

	std::array<double, 3>max{6.0, 6.0, 6.0};
	std::array<double, 3>min{-6.0, -6.0, -6.0};

	//------------------------
#ifdef _ROOT_AVAILABLE_

	TH3D hist_d("hist_d",   "3D Double Gaussian - Device",
			50, -6.0, 6.0,
			50, -6.0, 6.0,
			50, -6.0, 6.0 );

#endif //_ROOT_AVAILABLE_




	//device
	{

		hydra::multivector<
		      hydra::tuple<AxisX, AxisY, AxisZ>,
		      hydra::device::sys_t > dataset(nentries);


		auto start_d = std::chrono::high_resolution_clock::now();

		auto range = hydra::sample(dataset, min, max, Gaussians);

		auto end_d = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed_d = end_d - start_d;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| [Generation] Device Time (ms) ="<< elapsed_d.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

		std::cout <<std::endl;
		std::cout <<std::endl;
		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Sample > [" << i << "] :" << range.begin()[i] << std::endl;

		std::array<size_t, 3> nbins{50, 50, 50};

		hydra::SparseHistogram<double,3, hydra::device::sys_t> Hist_Data(nbins, min, max);

		start_d = std::chrono::high_resolution_clock::now();

		Hist_Data.Fill(range.begin(), range.end());

		end_d = std::chrono::high_resolution_clock::now();
		elapsed_d = end_d - start_d;


		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| [ Histograming ] Device Time (ms) ="<< elapsed_d.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

#ifdef _ROOT_AVAILABLE_

		{
			hydra::SparseHistogram<double,3, hydra::host::sys_t> _temp_hist=Hist_Data;

		for(size_t i=0;  i<50; i++){
					for(size_t j=0;  j<50; j++){
						for(size_t k=0;  k<50; k++){

							size_t bin[3]={i,j,k};

				          	hist_d.SetBinContent(i+1,j+1,k+1,
				          			 _temp_hist.GetBinContent(bin )  );
						}
					}
				}

		}
#endif //_ROOT_AVAILABLE_

	}



#ifdef _ROOT_AVAILABLE_
	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas_d("canvas_d" ,"Distributions - Device", 1000, 1000);
	hist_d.Draw("");
	hist_d.SetFillColor(9);


	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;



}

#endif /* SPARSE_HISTOGRAMING_INL_ */
