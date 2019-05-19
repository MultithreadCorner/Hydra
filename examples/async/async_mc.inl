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
 * async.inl
 *
 *  Created on: Oct 5, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ASYNC_INL_
#define ASYNC_INL_


#include <iostream>
#include <assert.h>
#include <time.h>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <limits>
#include <future>
#include <thread>

//command line arguments
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




#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>

//command line
#include <tclap/CmdLine.h>

//this lib
//backends
#include <hydra/omp/System.h>
#include <hydra/cuda/System.h>
#include <hydra/tbb/System.h>
#include <hydra/cpp/System.h>

#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/Random.h>
#include <hydra/Algorithm.h>
#include <hydra/Tuple.h>
#include <hydra/Distance.h>
#include <hydra/multiarray.h>
#include <hydra/SparseHistogram.h>
#include <hydra/DenseHistogram.h>
#include <hydra/Range.h>
/*-------------------------------------
 * Include classes from ROOT to fill
 * and draw histograms and plots.
 *-------------------------------------
 */
#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <TH1D.h>
#include <TH3D.h>
#include <TApplication.h>
#include <TCanvas.h>

#endif //_ROOT_AVAILABLE_

/**
 * \example async.inl
 * This example demonstrates how to generate Monte Carlo samples in parallel
 * deploying different backends asynchronously.
 */

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

	// some definitions
	double min   = -6.0;
	double max   =  6.0;
	double mean  =  0.0;
	double sigma =  1.5;
    size_t nbins =  100;

    auto GAUSSIAN3D =  [=] __hydra_dual__ (unsigned int n,double* x ){

    	double g = 1.0;

    	for(size_t i=0; i<3; i++){

    		double m2 = (x[i] - mean )*(x[i] - mean );
    		double s2 = (sigma+i/2.0)*(sigma+i/2.0);
    		g *= exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));
    	}

    	return g;
    };

    auto Gaussian3D = hydra::wrap_lambda( GAUSSIAN3D );


	//---------
	//generator
	hydra::Random<>
	Generator( std::chrono::system_clock::now().time_since_epoch().count() );


	//------------------------
	//histograms
#ifdef _ROOT_AVAILABLE_

	TH3D hist_cuda("hist_cuda", "Gaussian3D - Cuda",
			nbins, min, max,
			nbins, min, max,
			nbins, min, max );

	TH1D hist_omp("hist_omp",   "Gaussian - OMP",  nbins, min, max);

	TH1D hist_tbb("hist_tbb",   "Breit-Wigner - TBB",  nbins, min, max);

	TH1D hist_cpp("hist_cpp",   "Uniform - CPP",  nbins, min, max);

#endif //_ROOT_AVAILABLE_




	hydra::multiarray<double, 3, hydra::host::sys_t> dataset_cpu(nentries);
	hydra::multiarray<double, 3, hydra::device::sys_t> dataset_gpu(nentries);

	auto Histogram_CPP = std::async( std::launch::async, [=, &dataset_cpu, &Generator]  {

		Generator.Uniform(min, max, dataset_cpu.begin(0), dataset_cpu.end(0) );
		hydra::DenseHistogram<double, 1, hydra::host::sys_t > Histogram(nbins, min, max);
		Histogram.Fill( dataset_cpu.begin(0), dataset_cpu.end(0) );

		return  Histogram;
	} );

	auto Histogram_OMP = std::async( std::launch::async, [=, &dataset_cpu, &Generator]  {

		Generator.Gauss(0.0, 1.0, dataset_cpu.begin(1), dataset_cpu.end(1) );
		hydra::DenseHistogram<double, 1, hydra::host::sys_t> Histogram(nbins, min, max);
		Histogram.Fill( dataset_cpu.begin(1), dataset_cpu.end(1) );

		return  Histogram;
	} );

	auto Histogram_TBB = std::async( std::launch::async, [=, &dataset_cpu, &Generator]  {

		Generator.BreitWigner(0.0, 0.50, dataset_cpu.begin(2), dataset_cpu.end(2) );
		hydra::DenseHistogram<double, 1, hydra::host::sys_t> Histogram(nbins, min, max);
		Histogram.Fill( dataset_cpu.begin(2), dataset_cpu.end(2) );

		return  Histogram;
	} );

	auto Histogram_CUDA = std::async( std::launch::async, [=, &dataset_gpu, &Generator]  {

		std::array<double, 3> _max;
		std::array<double, 3> _min;
		std::array<size_t, 3>  _nbins;

		for(size_t i=0;i<3;i++){
			_max[i]   = max;
			_min[i]   = min;
			_nbins[i] = nbins;
		}

		auto range  = Generator.Sample( dataset_gpu.begin(), dataset_gpu.end(), _min, _max, Gaussian3D);
		hydra::SparseHistogram< double, 3,hydra::device::sys_t> Histogram(_nbins, _min, _max);
		Histogram.Fill( range.begin(), range.end() );

		return  Histogram;
	} );


#ifdef _ROOT_AVAILABLE_

	auto H_OMP = Histogram_OMP.get();
	auto H_TBB = Histogram_TBB.get();
	auto H_CPP = Histogram_CPP.get();

	for(size_t bin=0; bin < nbins; bin++){
		 hist_omp.SetBinContent( bin+1,  H_OMP[bin] );
		 hist_tbb.SetBinContent( bin+1,  H_TBB[bin] );
		 hist_cpp.SetBinContent( bin+1,  H_CPP[bin] );
	}

	auto H_CUDA = Histogram_CUDA.get();

	for(auto bin : H_CUDA){

		size_t index   = hydra::get<0>(bin);
		double content = hydra::get<1>(bin);


		int indexes[3];
		H_CUDA.GetIndexes(index, indexes );
		hist_cuda.SetBinContent(indexes[0]+1, indexes[1]+1, indexes[2]+1, content  );

	}
#endif //_ROOT_AVAILABLE_



#ifdef _ROOT_AVAILABLE_
	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas("canvas" ,"Distributions - Device", 1000, 1000);
	canvas.Divide(2,2);

	canvas.cd(1);
	hist_omp.Draw("hist");
	hist_omp.SetLineColor(2);

	canvas.cd(2);
	hist_tbb.Draw("hist");
	hist_tbb.SetLineColor(4);

	canvas.cd(3);
	hist_cpp.Draw("hist");
	hist_cpp.SetLineColor(3);
	hist_cpp.SetMinimum(0.0);

	canvas.cd(4);
	hist_cuda.Draw("");

	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;



}





#endif /* ASYNC_INL_ */
