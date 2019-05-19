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
 * zipping_iterables.inl
 *
 *  Created on: 02/07/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ZIPPING_ITERABLES_INL_
#define ZIPPING_ITERABLES_INL_


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
#include <hydra/Zip.h>
#include <hydra/Range.h>

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


	//device
	{
		std::cout << "=========================================="<<std::endl;
		std::cout << "|            <--- DEVICE --->            |"<<std::endl;
		std::cout << "=========================================="<<std::endl;


		hydra::device::vector<double> x( nentries , 1.0);
		hydra::device::vector<double> y( nentries , 2.0);
		hydra::device::vector<double> z( nentries , 3.0);

		auto zipped_range = hydra::zip(	hydra::range(0,  nentries), x,y,z);

		//print
		hydra::for_each(zipped_range, [] __hydra_dual__ ( hydra::tuple<long, double&, double&, double&> a){

			printf("row i = %d : x = %f, y = %f, z = %f\n", hydra::get<0>(a),hydra::get<1>(a), hydra::get<2>(a), hydra::get<3>(a));

		});

		//modify
		hydra::for_each(zipped_range, [] __hydra_dual__ ( hydra::tuple<long, double&, double&, double&> a){

			hydra::get<1>(a) = 10;
			hydra::get<2>(a) = 20;
			hydra::get<3>(a) = 30;
		});

		//print again
		hydra::for_each(zipped_range, [] __hydra_dual__ ( hydra::tuple<long, double&, double&, double&> a){

			printf("row i = %d : x = %f, y = %f, z = %f\n", hydra::get<0>(a),hydra::get<1>(a), hydra::get<2>(a), hydra::get<3>(a));

		});


	}//device


	return 0;
}





#endif /* ZIPPING_ITERABLES_INL_ */
