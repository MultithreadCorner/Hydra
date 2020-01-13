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
 * sobol_quasirandom.inl
 *
 *  Created on: 05/01/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SOBOL_QUASIRANDOM_INL_
#define SOBOL_QUASIRANDOM_INL_

#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>

//command line
#include <tclap/CmdLine.h>

//this lib
#include <hydra/Sobol.h>
#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Algorithm.h>
#include <hydra/Range.h>
#include <hydra/multiarray.h>
#include <hydra/device/System.h>

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

	hydra::sobol<2> eng;
	std::cout<< "According algorithm documentation, Sobol 2D output should be: \n"
			"x, y\n"
			"0.5, 0.5\n"
			"0.75, 0.25\n"
			"0.25, 0.75\n"
			"0.375, 0.375\n\n"
			<< std::endl;

	std::cout<< "hydra::sobol<2> [default seed] output (x, y) - (tx, ty) ns:" << std::endl;

	for(size_t i=0; i<10; ++i){

		auto start_x = std::chrono::high_resolution_clock::now();
		auto x = eng();
		auto stop_x = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::nano> elapsed_x = stop_x - start_x;
		auto start_y = std::chrono::high_resolution_clock::now();
		auto y = eng();
		auto stop_y = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::nano> elapsed_y = stop_y - start_y;

		std::cout<<"<"<<i << "> - ("
				<< (double)x/eng.Max()<< ", "
				<< (double)y/eng.Max()<< ") -- ( "
				<<  elapsed_x.count() << ", "
				<<  elapsed_y.count()<< ")"
				<< std::endl;

	}

	//fill a hydra::multiarray with 2D sobol numbers
	hydra::multiarray<double, 2, hydra::device::sys_t> sobol_numbers(nentries);

	auto sobol_sampler =  hydra::wrap_lambda( []__hydra_dual__(unsigned n, long  int* N) {

		hydra::sobol<2> eng;

		eng.discard((*N)*2);
		auto denominator=eng.Max();
		auto x= eng();
		auto y= eng();
		return hydra::make_tuple( (double) x/denominator, (double)y/denominator) ;

	});

	auto sobol_range= hydra::range(0, nentries ) | sobol_sampler ;



	auto start = std::chrono::high_resolution_clock::now();
	hydra::copy(sobol_range, sobol_numbers );
	auto stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed = stop - start;

	std::cout << "[Generation time (ms)]: " <<  elapsed.count() << std::endl;

	for(size_t i=0; i<10; ++i){
		std::cout<<"<"<<i << "> - "
				 << sobol_range[i]
				 << std::endl;
	}

return 0;
}




#endif /* SOBOL_QUASIRANDOM_INL_ */
