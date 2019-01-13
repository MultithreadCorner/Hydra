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
 * range_semantics.inl
 *
 *  Created on: 15/05/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef RANGE_SEMANTICS_INL_
#define RANGE_SEMANTICS_INL_

/**
 * \example range_semantics.inl
 *
 * This example shows how to use hydra range semantic
 * to perform lazy calculations.
 */

//
#include <iostream>
#include <algorithm>
//hydra stuff
#include <hydra/device/System.h>
#include <hydra/host/System.h>
#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Tuple.h>
#include <hydra/multiarray.h>
#include <hydra/Placeholders.h>
#include <hydra/Random.h>
#include <hydra/Algorithm.h>
#include <hydra/DenseHistogram.h>

//command line arguments
#include <tclap/CmdLine.h>

using namespace hydra::placeholders;

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


	// calculate the length of a n-dimensional vector
	auto length = hydra::wrap_lambda(
			[] __hydra_dual__ ( unsigned  n, double* component){

		double result =0;

		for(unsigned i=0; i<n; i++)
			result += component[i]* component[i];

		return ::sqrt(result);
	});

	// flag according distance to origin
	auto is_inside =  hydra::wrap_lambda(
			[] __hydra_dual__ (  unsigned n, double* radi){

		return radi[0]<1.0;
	});

	//device
	{
		std::cout << "=========================================="<<std::endl;
		std::cout << "|            <--- DEVICE --->            |"<<std::endl;
		std::cout << "=========================================="<<std::endl;


		hydra::multiarray<double, 3, hydra::device::sys_t> positions(nentries);

		hydra::Random<> Generator{};

		//generate random positions in a box
		for(size_t i=0; i<3; i++ ){

			Generator.SetSeed(i);
			Generator.Uniform(-1.5, 1.5, positions.begin(i), positions.end(i));

		}

		auto sorted_range = hydra::sort_by_key(positions, hydra::columns(positions, _0,_1 ) | length ) | is_inside;

		//hydra::for_each(positions, [] __hydra_dual__ ( hydra::tuple<double&, double&, double&> a){ a= hydra::tuple<double, double, double>{1,2,3}; } );

		for(auto i:sorted_range) std::cout << i << std::endl;

		hydra::for_each(positions, [] __hydra_dual__ ( hydra::tuple<double, double, double> a){

			printf("%f %f %f\n", hydra::get<0>(a),hydra::get<1>(a), hydra::get<2>(a));

		});

		std::array<double, 3> masses{0.13957061, 0.13957061,0.13957061};

		auto events =  hydra::phase_space_range(hydra::Vector4R(0.493677, 0.0, 0.0, 0.0),masses, 100000);

		auto invariant_mass = hydra::wrap_lambda(
				[]__hydra_dual__( hydra::tuple<double, hydra::Vector4R, hydra::Vector4R, hydra::Vector4R> event ){

			return (hydra::get<1>(event) + hydra::get<2>(event)).mass();
		});

		hydra::DenseHistogram<double,1, hydra::device::sys_t> Hist_Mass(100, masses[0]+masses[1], 0.493677 - masses[0]);

		hydra::for_each( Hist_Mass.Fill( events|invariant_mass ), [] __hydra_dual__ ( double a){
					printf("%f\n", a);
				});


	}//device


	return 0;
}



#endif /* RANGE_SEMANTICS_INL_ */
