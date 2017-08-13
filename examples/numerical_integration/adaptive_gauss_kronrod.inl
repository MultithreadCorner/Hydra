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
 * gauss_kronrod.inl
 *
 *  Created on: 17/07/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GAUSS_KRONROD_INL_
#define GAUSS_KRONROD_INL_



/**
 * @file
 * @example adaptive_gauss_kronrod.inl
 * This example show how to use the hydra::GaussKronrodAdaptiveQuadrature
 * self-adaptive numerical integration algorithm to calculate
 * the integral of a  Gaussian.
 */

#include <iostream>
#include <assert.h>
#include <time.h>
#include <string>
#include <chrono>
//command line arguments
#include <tclap/CmdLine.h>

//this lib
#include <hydra/Function.h>
#include <hydra/GaussKronrodAdaptiveQuadrature.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/host/System.h>
#include <hydra/device/System.h>



int main(int argv, char** argc)
{
	double max_error          = 0;

	try {

		TCLAP::CmdLine cmd("Command line arguments for vegas", '=');

		TCLAP::ValueArg<double> MaxErrorArg("e", "max-error", "Maximum error.", false, 1.0e-3, "double");
		cmd.add(MaxErrorArg);

		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		max_error  = MaxErrorArg.getValue();

	}
	catch (TCLAP::ArgException &e)
	{
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}

	//integration region limits
	double  min  = -6.0;
	double  max =  6.0;

    //Gaussian parameters
	double mean  = 0.0;
	double sigma = 1.0;


	// create functor using C++11 lambda
	auto GAUSSIAN = [=] __host__ __device__ (unsigned int n, double* x ){

		double m2 = (x[0] - mean )*(x[0] - mean );
		double s2 = sigma*sigma;
		double f = exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));

		return f;
	};

	//wrap the lambda
    auto gaussian = hydra::wrap_lambda(GAUSSIAN);

    //device
    {
    	// 61- degree quadrature
    	hydra::GaussKronrodAdaptiveQuadrature<61,10, hydra::device::sys_t> GKAQ61_d(min,  max, max_error);

    	auto start = std::chrono::high_resolution_clock::now();

    	auto result = GKAQ61_d.Integrate(gaussian);

    	auto end = std::chrono::high_resolution_clock::now();

    	std::chrono::duration<double, std::milli> elapsed = end - start;

    	std::cout << ">>>l [ Gauss-Kronrod 61 ]"<< std::endl;
    	std::cout << "Result: " << result.first << "  +-  " << result.second <<std::endl
    			<< " Time (ms): "<< elapsed.count() <<std::endl;
    }

    //host
    {
    	// 61- degree quadrature
    	hydra::GaussKronrodAdaptiveQuadrature<61,10, hydra::host::sys_t> GKAQ61_h(min,  max, max_error);

    	auto start = std::chrono::high_resolution_clock::now();

    	auto result = GKAQ61_h.Integrate(gaussian);

    	auto end = std::chrono::high_resolution_clock::now();

    	std::chrono::duration<double, std::milli> elapsed = end - start;

    	std::cout << ">>>l [ Gauss-Kronrod 61 ]"<< std::endl;
    	std::cout << "Result: " << result.first << "  +-  " << result.second <<std::endl
    			<< " Time (ms): "<< elapsed.count() <<std::endl;
    }

	return 0;


	}



#endif /* GAUSS_KRONROD_INL_ */
