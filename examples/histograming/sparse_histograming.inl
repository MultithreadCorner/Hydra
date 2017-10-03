/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016-2017 Antonio Augusto Alves Junior
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
#include <hydra/DenseHistogram.h>
#include <hydra/GenericRange.h>
/*-------------------------------------
 * Include classes from ROOT to fill
 * and draw histograms and plots.
 *-------------------------------------
 */
#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <THnSparse.h>
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
	double mean1   = -2.0;
	double sigma1  =  1.0;

	double _max    =  3.0;
	double _min    = -3.0;
	size_t _nbins  =  100;

	constexpr size_t N=10;

	/*
	 * In 10 dimensions, a dense histogram with 100 bins per dimension
	 * would allocates an array with 10^20 doubles, one for store the
	 * counting of each bin.
	 * Given each double has 16 bytes, 1.6 x 10^21 bytes or 1.6 x10^9 Terabytes!!!
	 * On the other hand, each entry has 10 doubles, to populate each bin with
	 * 10 doubles requires the storage or generation of 10^22 doubles.
	 * Sparse histograms will only store bins that are not empty.
	 */


	auto GAUSSIAN1 =  [=] __host__ __device__ (unsigned int n,double* x ){

		double g = 1.0;

		for(size_t i=0; i<10; i++){

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

		for(size_t i=0; i<10; i++){

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
	hydra::Random<>
	Generator( std::chrono::system_clock::now().time_since_epoch().count() );

	std::array<double, N>max;
	std::array<double, N>min;
    std::array<size_t, N> nbins;

	for(size_t i=0;i<N;i++){
		max[i]=_max;
		min[i]=_min;
		nbins =_nbins;
	}

	//------------------------
#ifdef _ROOT_AVAILABLE_
	THnSparseD histo_h("histo_d", "Histogram Host", N, _nbins, min, max);
	THnSparseD histo_d("histo_d", "Histogram Device", N, _nbins, min, max);
#endif //_ROOT_AVAILABLE_



	typedef hydra::multiarray<N, double, hydra::device::sys_t> dataset_d;
	typedef hydra::multiarray<N, double, hydra::host::sys_t> dataset_h;

	//device
	{

		dataset_d data_d(nentries);

		auto start_d = std::chrono::high_resolution_clock::now();

		auto range = Generator.Sample(data_d.begin(),  data_d.end(), min, max, gaussians);

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



		hydra::SparseHistogram<N, double> Hist_Data(nbins, min, max);

		start_d = std::chrono::high_resolution_clock::now();

		Hist_Data.Fill(range.begin(), range.end());

		end_d = std::chrono::high_resolution_clock::now();

		elapsed_d = end_d - start_d;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| [ Histograming ] Device Time (ms) ="<< elapsed_d.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

#ifdef _ROOT_AVAILABLE_
		for(size_t i=0;  i<50; i++){
			for(size_t j=0;  j<50; j++){
				for(size_t k=0;  k<50; k++){

					size_t bin[3]{i,j,k};

					hist_d.SetBinContent(i+1,j+1,k+1,
							Hist_Data.GetBinContent(bin )  );
				}
			}
		}
#endif //_ROOT_AVAILABLE_

	}

	//host
	{

		dataset_h data_h(nentries);

		auto start_d = std::chrono::high_resolution_clock::now();
		auto range = Generator.Sample(data_h.begin(),  data_h.end(), min, max, gaussians);
		auto end_d = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_d = end_d - start_d;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| [Generation] Host Time (ms) ="<< elapsed_d.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;


		std::cout <<std::endl;
		std::cout <<std::endl;

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Sample > [" << i << "] :" << range.begin()[i] << std::endl;

		std::array<size_t, 3> nbins{50, 50, 50};

		hydra::DenseHistogram<3, double> Hist_Data(nbins, min, max);

		start_d = std::chrono::high_resolution_clock::now();

		Hist_Data.Fill(range.begin(), range.end());

		end_d = std::chrono::high_resolution_clock::now();
		elapsed_d = end_d - start_d;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| [ Histograming ] Host Time (ms) ="<< elapsed_d.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

#ifdef _ROOT_AVAILABLE_
		for(size_t i=0;  i<50; i++){
			for(size_t j=0;  j<50; j++){
				for(size_t k=0;  k<50; k++){

					size_t bin[3]{i,j,k};

		          	hist_h.SetBinContent(i+1,j+1,k+1,
		          			Hist_Data.GetBinContent(bin )  );
				}
			}
		}
#endif //_ROOT_AVAILABLE_


	}


#ifdef _ROOT_AVAILABLE_
	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas_d("canvas_d" ,"Distributions - Device", 1000, 1000);
	hist_d.Draw("hist");
	hist_d.SetFillColor(9);

	//draw histograms
	TCanvas canvas_h("canvas_h" ,"Distributions - Host", 1000, 1000);
	hist_h.Draw("hist");
	hist_h.SetFillColor(9);

	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;



}




#endif /* SPARSE_HISTOGRAMING_INL_ */
