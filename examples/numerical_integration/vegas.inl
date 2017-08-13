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
 * vegas.inl
 *
 *  Created on: 16/07/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef VEGAS_INL_
#define VEGAS_INL_

/**
 * @file
 * @example vegas.inl
 * This example show how to use the hydra::Vegas
 * numerical integration algorithm to calculate
 * the integral of a five dimensional Gaussian.
 */

#include <iostream>
#include <assert.h>
#include <time.h>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <limits>


//command line arguments
#include <tclap/CmdLine.h>

//this lib
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/VegasState.h>
#include <hydra/Vegas.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/host/System.h>
#include <hydra/device/System.h>



int main(int argv, char** argc)
{

	size_t  calls             = 0;
	size_t  iterations        = 0;
	double max_error          = 0;

	try {

		TCLAP::CmdLine cmd("Command line arguments for vegas", '=');

		TCLAP::ValueArg<size_t> NCallsArg("n", "number-of-calls", "Number of call.", true, 5000, "size_t");
		cmd.add(NCallsArg);

		TCLAP::ValueArg<double> MaxErrorArg("e", "max-error", "Maximum error.", false, 1.0e-3, "double");
		cmd.add(MaxErrorArg);

		TCLAP::ValueArg<size_t> IterationsArg("i", "max-iterations", "Maximum maximum number of iterations.",false, 10, "size_t");
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

	//number of dimensions (user can change it)
	constexpr size_t N = 5;

	//integration region limits
	double  min[N];
	double  max[N];

    //5D Gaussian parameters
	double mean  = 0.0;
	double sigma = 1.0;

	//set Gaussian parameters and
	//integration region limits
	for(size_t i=0; i< N; i++){
		min[i]   = -6.0;
		max[i]   =  6.0;
	}

	// create functor using C++11 lambda
	auto GAUSSIAN = [=] __host__ __device__ (unsigned int n, double* x ){

		double g = 1.0;
		double f = 0.0;

		for(size_t i=0; i<N; i++){

			double m2 = (x[i] - mean )*(x[i] - mean );
			double s2 = sigma*sigma;
			f = exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));
			g *= f;
		}

		return g;
	};

	//wrap the lambda
    auto gaussian = hydra::wrap_lambda(GAUSSIAN);

    //device
    {
    	//----------------------------------------------------------------------
    	//Vegas State_d hold the resources for performing the integration
    	hydra::VegasState<N,  hydra::device::sys_t> State_d(min, max);
    	State_d.SetVerbose(-2);
    	State_d.SetAlpha(1.5);
    	State_d.SetIterations( iterations );
    	State_d.SetUseRelativeError(1);
    	State_d.SetMaxError( max_error );
    	State_d.SetCalls( calls );
    	State_d.SetTrainingCalls( calls/10 );
    	State_d.SetTrainingIterations(2);

    	//vegas integrator
    	hydra::Vegas<N,  hydra::device::sys_t > Vegas_d(State_d);

    	auto start_vegas = std::chrono::high_resolution_clock::now();
    	auto result = Vegas_d.Integrate(gaussian);
    	auto end_vegas = std::chrono::high_resolution_clock::now();
    	std::chrono::duration<double, std::milli> elapsed_vegas = end_vegas - start_vegas;
    	std::cout << std::endl;
    	std::cout << "----------------- Device ----------------"<< std::endl;
    	std::cout << ">>> [Vegas]: Gaussian<"<< N << ">" << std::endl;
    	std::cout << "Result: "    << Vegas_d.GetState().GetResult() << " +/- " << Vegas_d.GetState().GetSigma() <<std::endl
				  << "Time (ms): " << elapsed_vegas.count() <<std::endl;
    	std::cout << "-----------------------------------------"<< std::endl;

    }

    //host
    {
    	//----------------------------------------------------------------------
    	//Vegas State_d hold the resources for performing the integration
    	hydra::VegasState<N, hydra::host::sys_t> State_h(min, max);
    	State_h.SetVerbose(-2);
    	State_h.SetAlpha(1.5);
    	State_h.SetIterations( iterations );
    	State_h.SetUseRelativeError(1);
    	State_h.SetMaxError( max_error );
    	State_h.SetCalls( calls );
    	State_h.SetTrainingCalls( calls/10 );
    	State_h.SetTrainingIterations(2);

    	//vegas integrator
    	hydra::Vegas<N,  hydra::host::sys_t > Vegas_h(State_h);

    	auto start_vegas = std::chrono::high_resolution_clock::now();
    	Vegas_h.Integrate(gaussian);
    	auto end_vegas = std::chrono::high_resolution_clock::now();
    	std::chrono::duration<double, std::milli> elapsed_vegas = end_vegas - start_vegas;
    	std::cout << std::endl;
    	std::cout << "----------------- Host ----------------"<< std::endl;
    	std::cout << ">>> [Vegas]: Gaussian<" << N << ">" << std::endl;
    	std::cout << "Result: "   << Vegas_h.GetState().GetResult() << " +/- " << Vegas_h.GetState().GetSigma() <<std::endl
				  << "Time (ms): "<< elapsed_vegas.count() << std::endl;
    	std::cout << "-----------------------------------------"<< std::endl;

    }

	return 0;


	}



#endif /* VEGAS_INL_ */
