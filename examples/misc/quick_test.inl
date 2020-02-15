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
 * quick_test.inl
 *
 *  Created on: 13/02/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef QUICK_TEST_INL_
#define QUICK_TEST_INL_

#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>

//command line
#include <tclap/CmdLine.h>

//hydra
#include <hydra/host/System.h>
#include <hydra/device/System.h>
#include <hydra/Lambda.h>
#include <hydra/multivector.h>
#include <hydra/Parameter.h>

using namespace hydra::arguments;

declarg(xvar, double)
declarg(yvar, double)
declarg(zvar, double)

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
		std::cerr << " error: "  << e.error()
				  << " for arg " << e.argId()
				  << std::endl;
	}

	auto lambda = []__hydra_dual__(xvar x, yvar y)
			{

			printf(" X = %f Y = %f\n", x(), y());

			return;
		   };

	auto plambda = []__hydra_dual__(size_t n, hydra::Parameter* pars, xvar x, yvar y)
				{

				printf(" X = %f Y = %f\n", x(), y());

				return;
			   };


	auto wlambda = hydra::wrap_lambda(lambda);

	hydra::multivector<hydra::tuple<xvar,yvar, zvar>,
			            hydra::device::sys_t> dataset(nentries, hydra::make_tuple(1.0, 2.0, 3.0));

	std::cout << " : " <<  std::is_convertible<hydra::tuple<double>,
			hydra::tuple<hydra::detail::FunctionArgument<xvar, typename xvar::value_type>>>::value<< std::endl;
	wlambda(1.0, 2.0);
    for(auto x:dataset)
    	wlambda(x);

	return 0;
}



#endif /* QUICK_TEST_INL_ */
