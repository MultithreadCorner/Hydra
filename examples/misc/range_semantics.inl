/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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



#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>

#include <hydra/device/System.h>
#include <hydra/host/System.h>
#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Tuple.h>
#include <hydra/multivector.h>
#include <hydra/multiarray.h>
#include <hydra/Placeholders.h>
#include <hydra/Random.h>

//command line
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


	auto length = hydra::wrap_lambda(
			[] __hydra_dual__ ( unsigned n, double* component){

		double result =0;

		for(unsigned i=0; i<n; i++)
			result += component[i]* component[i];

		return ::sqrt(result);
	});

	auto is_inside =  hydra::wrap_lambda(
			[] __hydra_dual__ (  unsigned n, double* radi){

		return radi[0]<1.0;
	});

	//device
	{
		std::cout << "=========================================="<<std::endl;
		std::cout << "|            <--- DEVICE --->            |"<<std::endl;
		std::cout << "=========================================="<<std::endl;


		hydra::multiarray<double, 3, hydra::device::sys_t> positions_d(nentries);

		hydra::Random<> Generator{};

		//generate positions
		for(size_t i=0; i<3; i++ ){
			Generator.SetSeed(i);
			Generator.Uniform(-1.0, 1.0, positions_d.begin(i), positions_d.end(i));

		}

		auto range =  positions_d | length | is_inside ;

		//print elements
		std::cout<< std::endl << "________________________________________________________________________________" << std::endl<< std::endl;
		for(size_t i=0; i<nentries; i++ )
			if(range[i]) std::cout  << "Inside sphere : "<< positions_d[i]<< std::endl;
			else std::cout  << "Outside sphere : "<< positions_d[i]<< std::endl;


	}//device


	return 0;
}



#endif /* RANGE_SEMANTICS_INL_ */
