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
 * phsp_averaging_functor.inl
 *
 *  Created on: 12/07/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PHSP_AVERAGING_FUNCTOR_INL_
#define PHSP_AVERAGING_FUNCTOR_INL_



/**
 * @example phsp_averaging_functor.inl
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
 * algorithms for
 *--------------------------------
 */
#include <hydra/Types.h>
#include <hydra/Vector4R.h>
#include <hydra/PhaseSpace.h>
#include <hydra/Events.h>
#include <hydra/Chain.h>
#include <hydra/Evaluate.h>
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Copy.h>
#include <hydra/Tuple.h>
#include <hydra/host/System.h>
#include <hydra/device/System.h>

/*-------------------------------------
 * Include classes from ROOT to fill
 * and draw histograms and plots.
 *-------------------------------------
 */
#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <TH1D.h>
#include <TF1.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TString.h>
#include <TStyle.h>

#endif //_ROOT_AVAILABLE_



int main(int argv, char** argc)
{


	size_t  nentries   = 0; // number of events to generate, to be get from command line
	double B0_mass    = 5.27955;   // B0 mass
	double Jpsi_mass  = 3.0969;    // J/psi mass
	double K_mass     = 0.493677;  // K+ mass
	double pi_mass    = 0.13957061;// pi mass


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

	//C++11 lambda for cosine of helicity angle Kpi
	auto COSHELANG = [] __host__ __device__ (unsigned int n,  hydra::Vector4R *fvectors )
	{
		hydra::Vector4R p1 = fvectors[0];
		hydra::Vector4R p2 = fvectors[1];
		hydra::Vector4R p3 = fvectors[2];

		hydra::Vector4R p = p1 + p2 + p3;
		hydra::Vector4R q = p2 + p3;

		double pd = p * p2;
		double pq = p * q;
		double qd = q * p2;
		double mp2 = p.mass2();
		double mq2 = q.mass2();
		double md2 = p2.mass2();

		return (pd * mq2 - pq * qd)
				/ sqrt((pq * pq - mq2 * mp2) * (qd * qd - mq2 * md2));
	};

	auto cosTheta = hydra::wrap_lambda(COSHELANG);

	hydra::Vector4R B0(B0_mass, 0.0, 0.0, 0.0);
	double masses[3]{Jpsi_mass, K_mass, pi_mass };

	// Create PhaseSpace object for B0-> K pi J/psi
	hydra::PhaseSpace<3> phsp(B0_mass, masses);

	//device
	{
	auto start = std::chrono::high_resolution_clock::now();

	auto device_result = phsp.AverageOn(hydra::device::sys, B0 , cosTheta, nentries) ;

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
	}



	//host
	{
		auto start = std::chrono::high_resolution_clock::now();

		auto host_result = phsp.AverageOn(hydra::host::sys, B0 , cosTheta, nentries) ;

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "----------------- Host ----------------"<< std::endl;
		std::cout << "|< cos(theta_K) >(B0 -> J/psi K pi): "
				<< host_result.first
				<< " +- "
				<< host_result.second
				<< std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;
	}


	return 0;
}




#endif /* PHSP_AVERAGING_FUNCTOR_INL_ */
