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
 * HydraFitExample.cu
 *
 *  Created on: Jun 21, 2016
 *      Author: Antonio Augusto Alves Junior
 */

#include <iostream>
#include <assert.h>
#include <time.h>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <limits>
#include <future>

//OpenMP
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
#include <omp.h>
#include <thread>
#endif

//command line arguments
#include <tclap/CmdLine.h>

//this lib
#include <hydra/Types.h>
#include <hydra/Vector4R.h>
#include <hydra/Containers.h>
#include <hydra/Function.h>
#include <hydra/VegasState.h>
#include <hydra/Vegas.h>
#include <hydra/Parameter.h>

//root
#include <TROOT.h>
#include <TH1D.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TString.h>
#include <TLegend.h>

#include <hydra/host/System.h>
#include <hydra/device/System.h>
#include <hydra/omp/System.h>
#include <hydra/cpp/System.h>
#include <hydra/cuda/System.h>
#include <hydra/tbb/System.h>


#include <examples/Gauss.h>
#include <examples/Exp.h>



using namespace std;
using namespace hydra;
using namespace examples;

/**
 * @file
 * @example HydraFitExample.cu
 * @brief HydraFitExample take parameters from the command line, fill a range with random numbers sampled from
 * the model and perform a extended likelihood fit in parallel using the OpenMP backend.
 * @param -c (--combined-minimizer):  Use Migrad + Simplex for minimization
 * @param -i=<double> (--max-iterations=<double>) : Maximum number of iterations for migrad and minimize call.
 * @param -t=<double> (--tolerance=<double>) : Tolerance parameter for migrad and minimize call.
 * @param -n=<long> (--number-of-events=<long>) (required):  Number of events for each component.
 *
 * Usage:
 * ./Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Fit  [-c] [-i=<double>]
 *                                      [-t=<double>] -n=<long> [--]
 *                                      [--version] [-h]
 *
 * For example, the command below:
 * ```
 * ./Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Fit -n=1000000
 * ```
 * will print some stuff to standard output and produce the plot:
 *
 * @image html Fit_CUDA.png
 */


/**
 * @file
 * @brief HydraFitExample take parameters from the command line, fill a range with random numbers sampled from
 * the model and perform a extended likelihood fit in parallel using the OpenMP backend.
 * @param -c (--combined-minimizer):  Use Migrad + Simplex for minimization
 * @param -i=<double> (--max-iterations=<double>) : Maximum number of iterations for migrad and minimize call.
 * @param -t=<double> (--tolerance=<double>) : Tolerance parameter for migrad and minimize call.
 * @param -n=<long> (--number-of-events=<long>) (required):  Number of events for each component.
 *
 * Usage:
 * ./Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Fit  [-c] [-i=<double>]
 *                                      [-t=<double>] -n=<long> [--]
 *                                      [--version] [-h]
 *
 * For example, the command below:
 * ```
 * ./Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Fit -n=1000000
 * ```
 * will print some stuff to standard output and produce the plot:
 *
 * @image html Fit_CUDA.png
 */

GInt_t main(int argv, char** argc)
{

	size_t  calls              = 0;
	size_t  iterations         = 0;
	GReal_t max_error          = 0;

	try {

		TCLAP::CmdLine cmd("Command line arguments for HydraRandomExample", '=');

		TCLAP::ValueArg<size_t> NCallsArg("n", "number-of-calls",
				"Number of call.",
				true, 5000, "size_t");
		cmd.add(NCallsArg);

		TCLAP::ValueArg<GReal_t> MaxErrorArg("e", "maximum-error",
				"Maximum error.",
				false, 1.0e-3, "double");
		cmd.add(MaxErrorArg);

		TCLAP::ValueArg<size_t> IterationsArg("i", "max-iterations",
				"Maximum maximum number of iterations.",
				false, 10, "size_t");
		cmd.add(IterationsArg);


		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		calls      = NCallsArg.getValue();
		iterations = IterationsArg.getValue();
		max_error  = MaxErrorArg.getValue();

	}
	catch (TCLAP::ArgException &e)
	{
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}


	constexpr size_t N = 5;


	//------------------------------------
	//parameters
	//------------------------------------

	std::string  Mean_s[N];
	std::string Sigma_s[N];
	GUInt_t  Position_p[N];
	Parameter    Mean_p[N];
	Parameter   Sigma_p[N];

    //-------------------------------------------
	//range of the analysis
	std::array<GReal_t, N>  _min;
	std::array<GReal_t, N>  _max;

	for(size_t i=0; i< N; i++){

		    _min[i] = -6.0;
		    _max[i] =  6.0;
	 Position_p[i] = i;
		 Mean_s[i] = "mean_"  ;
		 Mean_s[i] += std::to_string(i);
		Sigma_s[i] = "sigma_" ;
		Sigma_s[i] += std::to_string(i);
		 Mean_p[i].Name(Mean_s[i]).Value(0.0) .Error(0.0001).Limits( -5.0, 5.0);
		Sigma_p[i].Name(Sigma_s[i]).Value(1.0) .Error(0.0001).Limits( 0.5, 1.5);
	}

	//----------------------------------------------------------------------
	// create functor
	//------------------------------------

	GaussN<N> Gaussian(Mean_p, Sigma_p, Position_p, 1);
    Gaussian.PrintRegisteredParameters();

	//----------------------------------------------------------------------
	// Create all in CPP back-end first

	VegasState<N,  hydra::cpp::sys_t> State_CPP(_min, _max);

	State_CPP.SetVerbose(-2);
	State_CPP.SetAlpha(1.5);
	State_CPP.SetIterations( iterations );
	State_CPP.SetUseRelativeError(1);
	State_CPP.SetMaxError( max_error );
	State_CPP.SetCalls( calls );
	State_CPP.SetTrainingCalls( calls/10 );
	State_CPP.SetTrainingIterations(1);

	Vegas<N,  hydra::cpp::sys_t > Vegas_CPP(State_CPP);

	//---------------------------------------------------------------------
	// copy all to other Back-end
	Vegas<N, hydra::omp::sys_t> Vegas_OMP(Vegas_CPP);
	Vegas<N, hydra::cuda::sys_t> Vegas_CUDA(Vegas_CPP);
	Vegas<N, hydra::tbb::sys_t> Vegas_TBB(Vegas_CPP);

	//typedef std::future< std::pair<GReal_t, GReal_t> >  result_t;

	auto	result_CPP = std::async( std::launch::async, [=, &Vegas_CPP]
	   {
		auto r = Vegas_CPP.Integrate(Gaussian);
	return  r;
	    }
	);

	auto result_CUDA = std::async( std::launch::async, [=,&Vegas_CUDA]
	     {
		auto r = Vegas_CUDA.Integrate(Gaussian);
		return   r;
	     }
	);

	auto result_TBB = std::async( std::launch::async, [=,&Vegas_TBB]
		     {
		auto r = Vegas_TBB.Integrate(Gaussian);
		return    r;

		     }
		);

	auto	result_OMP = std::async( std::launch::async, [=,&Vegas_OMP]
			     {
		auto r =Vegas_OMP.Integrate(Gaussian);
		return    r;

			     }
			);

	std::cout <<std::endl;
	std::cout << std::setprecision(15)<< ">>>[CPP]:\t" <<  result_CPP .get()  <<std::endl;
	std::cout << std::setprecision(15)<< ">>>[CUDA]:\t" <<  result_CUDA .get()  <<std::endl;
	std::cout << std::setprecision(15)<< ">>>[OMP]:\t" <<  result_OMP.get()  <<std::endl;
	std::cout << std::setprecision(15)<< ">>>[TBB]:\t" <<  result_TBB.get()  <<std::endl;



	/*
	 cout <<endl;
		cout << ">>> [Vegas]: Gaussian<"<< N << ">" << " using "<< nt <<" OMP threads" <<endl;
		cout << "Result: " << Vegas_d.GetState().GetResult()
					 << " +/- "    << Vegas_d.GetState().GetSigma() <<std::endl
					 << "Time (ms): "<< elapsed_vegas.count() <<std::endl;

	 */

	return 0;


	}

