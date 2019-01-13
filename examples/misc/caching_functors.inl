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
 * caching_functors.inl
 *
 *  Created on: 23/04/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CACHING_FUNCTORS_INL_
#define CACHING_FUNCTORS_INL_


#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>

#include <hydra/device/System.h>
#include <hydra/host/System.h>
#include <hydra/Cache.h>
#include <hydra/Vector4R.h>
#include <hydra/PhaseSpace.h>
#include <hydra/functions/BreitWignerLineShape.h>
#include <hydra/functions/CosHelicityAngle.h>
#include <hydra/functions/ZemachFunctions.h>
#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/Parameter.h>
//command line
#include <tclap/CmdLine.h>


int main(int argv, char** argc)
{

	size_t  nentries  = 0; // number of events to generate, to be get from command line
	double B0_mass    = 5.27955;   // B0 mass
	double Jpsi_mass  = 3.0969;    // J/psi mass
	double K_mass     = 0.493677;  // K+ mass
	double pi_mass    = 0.13957061;// pi mass

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

	// Mother particle
	hydra::Vector4R B0(B0_mass, 0.0, 0.0, 0.0);

	// Create PhaseSpace object for B0-> K pi J/psi
	hydra::PhaseSpace<3> phsp{Jpsi_mass, K_mass, pi_mass};

	//allocate memory to hold the final states particles
	hydra::Decays<3, hydra::device::sys_t > Events_d(nentries);

	auto start = std::chrono::high_resolution_clock::now();

	//generate the final state particles
	phsp.Generate(B0, Events_d.begin(), Events_d.end());

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> elapsed = end - start;

	//output
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "-----------------------------------------"<< std::endl;
	std::cout << "| B0 -> J/psi K pi"                       << std::endl;
	std::cout << "| Number of events :"<< nentries          << std::endl;
	std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
	std::cout << "-----------------------------------------"<< std::endl;

	auto mass    = hydra::Parameter::Create("MASS_KST_892").Value(0.89555);
	auto width   = hydra::Parameter::Create("WIDTH_KST_892").Value(0.0473);



	 hydra::BreitWignerLineShape<hydra::PWave> breit_wigner(mass, width,
			 B0_mass, K_mass, pi_mass, Jpsi_mass, 1.5);

	 auto line_shape = hydra::wrap_lambda(
			 [=]__hydra_dual__(unsigned int n, hydra::Vector4R* p){

		 hydra::Vector4R p1 = p[0];
		 hydra::Vector4R p2 = p[1];
		 hydra::Vector4R p3 = p[2];

		 double theta = (p2+p3).mass();


         return breit_wigner(theta);
	 });

	 auto angular_distribution = hydra::wrap_lambda(
			 []__hydra_dual__(unsigned int n, hydra::Vector4R* p){

		 hydra::Vector4R p1 = p[0];
		 hydra::Vector4R p2 = p[1];
		 hydra::Vector4R p3 = p[2];

		 hydra::CosHelicityAngle fCosDecayAngle;
		 hydra::ZemachFunction<hydra::PWave> fAngularDist;

		 double theta = fCosDecayAngle( (p1+p2+p3), (p1+p2), p1 );

         return fAngularDist(theta);
	 });

	 //
	 std::cout << "Before cache building:" << std::endl;
	 std::cout << "Angular cache index: "<< angular_distribution.GetCacheIndex() << std::endl;
	 std::cout << "Breit-Wigner cache index: "<< line_shape.GetCacheIndex() << std::endl;
	 auto particles        = Events_d.GetUnweightedDecays();

	auto cache = hydra::make_cache(hydra::device::sys, particles.begin(), particles.end(),
			 angular_distribution, line_shape);


		 //
	 std::cout << "After cache building:" << std::endl;
	 std::cout << "Angular cache: "<< angular_distribution.GetCacheIndex() << std::endl;
	 std::cout << "Breit-Wigner cache: "<< line_shape.GetCacheIndex() << std::endl;

	 std::cout << "Dumping cache (10 first entries):" << std::endl;
	 for(size_t i=0; i<10; i++)

		 std::cout <<  cache.begin()[i] << std::endl;
	 return 0;
}

#endif /* CACHING_FUNCTORS_INL_ */
