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
 * phsp_averaging_functor.inl
 *
 *  Created on: 12/07/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PHSP_AVERAGING_FUNCTOR_INL_
#define PHSP_AVERAGING_FUNCTOR_INL_



/**
 * \example phsp_averaging_functor.inl
 *
 * This example shows how to use the Hydra's
 * phase space Monte Carlo algorithms to calculate the
 * average value and corresponding variance of a functor
 * over the phase space of the decay B0 -> J/psi K pi.
 */


/*---------------------------------
 * std
 * ---------------------------------
 */
#include <iostream>
#include <assert.h>
#include <time.h>
#include <vector>
#include <array>
#include <chrono>

/*---------------------------------
 * command line arguments
 *---------------------------------
 */
#include <tclap/CmdLine.h>

/*---------------------------------
 * Include hydra classes and
 * algorithms for/Containers.h
 *--------------------------------
 */
#include <hydra/Types.h>
#include <hydra/Vector4R.h>
#include <hydra/PhaseSpace.h>
#include <hydra/Function.h>
#include <hydra/Lambda.h>
#include <hydra/Algorithm.h>
#include <hydra/Tuple.h>
#include <hydra/host/System.h>
#include <hydra/device/System.h>
#include <hydra/Decays.h>
#include <hydra/DenseHistogram.h>
#include <hydra/Range.h>
#include <hydra/functions/CosHelicityAngle.h>


// Daughter particles

declarg(A, hydra::Vector4R)
declarg(B, hydra::Vector4R)
declarg(C, hydra::Vector4R)

//---------------------------
using namespace hydra::arguments;


int main(int argv, char** argc)
{


	size_t  nentries   = 0; // number of events to generate, to be get from command line

	double P_mass = 5.27955;
	double A_mass = 3.0969;
	double B_mass = 0.493677;
	double C_mass = 0.13957061;


	try {

		TCLAP::CmdLine cmd("Command line arguments for PHSP B0 -> J/psi K pi", '=');

		TCLAP::ValueArg<size_t> NArg("n",
				"nevents",
				"Number of events to generate. Default is [ 10e6 ].",
				true, 10e6, "unsigned long");
		cmd.add(NArg);

		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		nentries       = NArg.getValue();

	}
	catch (TCLAP::ArgException &e)  {
		std::cerr << "error: " << e.error() << " for arg " << e.argId()
																<< std::endl;
	}


	auto cosTheta = hydra::wrap_lambda(  [] __hydra_dual__ ( A a, B b, C c) {
		auto coshelang = hydra::CosHelicityAngle();

		return coshelang(a+b+c, b+c, c);
	});

	hydra::Vector4R Parent(P_mass, 0.0, 0.0, 0.0);

	double masses[3]{A_mass, B_mass, C_mass };

	// Create PhaseSpace object for B0-> K pi J/psi
	hydra::PhaseSpace<3> phsp{P_mass, masses};

	//device
	{
	auto start = std::chrono::high_resolution_clock::now();

	auto device_result = phsp.AverageOn(hydra::device::sys, Parent, cosTheta, nentries) ;

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> elapsed = end - start;

	//output
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "----------------- Device ----------------"<< std::endl;
	std::cout << "|< cos(theta_K) >(B0 -> J/psi K pi): "
			  << device_result.first
			  << " +- "
			  << device_result.second
			  << std::endl;
	std::cout << "| Number of events :"<< nentries          << std::endl;
	std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
	std::cout << "-----------------------------------------"<< std::endl;

	}//device





	return 0;
}




#endif /* PHSP_AVERAGING_FUNCTOR_INL_ */
