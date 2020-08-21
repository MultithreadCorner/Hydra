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
#include <hydra/Parameter.h>
#include <hydra/detail/Compose.h>
#include <hydra/detail/Divide.h>
#include <hydra/detail/Minus.h>
#include <hydra/detail/Multiply.h>
#include <hydra/detail/Sum.h>
#include <hydra/functions/Gaussian.h>



#include <hydra/detail/external/hydra_thrust/random.h>

#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <TH1D.h>
#include <TApplication.h>
#include <TCanvas.h>

#endif //_ROOT_AVAILABLE_


using namespace hydra::arguments;

declarg(xvar, double)
declarg(yvar, double)

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


    auto data = hydra::device::vector< double>(10, .0);

	//Parameters
	auto mean1   = hydra::Parameter::Create("mean1"  ).Value(-1.0);
	auto mean2   = hydra::Parameter::Create("mean2"  ).Value( 1.0);
	auto sigma   = hydra::Parameter::Create("sigma" ).Value(1.0);
	auto factor  = hydra::Parameter::Create("factor").Value(1.0);

	//Gaussian distribution
	auto gauss1 = hydra::Gaussian<xvar>(mean1, sigma);
	//LogNormal distribution
	auto gauss2 = hydra::Gaussian<yvar>(mean2, sigma);
	//

	auto combiner = hydra::wrap_lambda( [] __hydra_dual__ (unsigned int npar, const hydra::Parameter* params, double x, double y) {

		printf("Gauss1 %f , Gauss2 %f\n", x, y );
		return x + params[0]*y;
	}, factor);

	auto fcomposed = hydra::compose( combiner, gauss1, gauss2);
	auto fdivided  = hydra::divide( gauss1, gauss2  );
	auto fminus    = hydra::minus( gauss1, gauss2  );
	auto fmultiply = hydra::multiply( gauss1, gauss2 , gauss2  );
	auto fsum      = hydra::sum( gauss1, gauss2 , gauss2  );

    for(size_t i=0; i< data.size(); ++i)
    {
    	std::cout << "fcomposed: " << fcomposed(xvar(-1.0), yvar(1.0)) << std::endl;
    	std::cout << "fdivided:  " << fdivided(xvar(-1.0), yvar(1.0))  << std::endl;
    	std::cout << "fminus:    " << fminus(xvar(-1.0), yvar(1.0))    << std::endl;
    	std::cout << "fmultiply: " << fmultiply(xvar(-1.0), yvar(1.0)) << std::endl;
    	std::cout << "fsum:      " << fsum(xvar(-1.0), yvar(1.0)) << std::endl;

    }



	return 0;
}



#endif /* QUICK_TEST_INL_ */
