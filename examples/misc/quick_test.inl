/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2022 Antonio Augusto Alves Junior
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
#include <typeinfo>
#include <type_traits>
//command line
#include <tclap/CmdLine.h>

//hydra
#include <hydra/host/System.h>
#include <hydra/device/System.h>
#include <hydra/Lambda.h>
#include <hydra/Parameter.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/functions/Gaussian.h>



#include <hydra/detail/external/hydra_thrust/random.h>

#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <TH1D.h>
#include <TApplication.h>
#include <TCanvas.h>

#endif //_ROOT_AVAILABLE_


using namespace hydra::arguments;

declarg(_u, double)
declarg(_v, double)
declarg(_x, double)
declarg(_y, double)


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
	auto mean_x   = hydra::Parameter::Create("mean_x"  ).Value(-0.5);
	auto mean_y   = hydra::Parameter::Create("mean_y"  ).Value(0.6);
	auto mean_u   = hydra::Parameter::Create("mean_u"  ).Value(-0.5);
	auto mean_v   = hydra::Parameter::Create("mean_v"  ).Value(0.6);
	auto sigma    = hydra::Parameter::Create("sigma"   ).Value(1.0);
	auto exponent = hydra::Parameter::Create("exponent"   ).Value(2.0);

	//build the expression
	// [ ( G(x)-G(y) ) / ( G(x) + G(y) ) +  ( G(u)-G(v) ) / ( G(u) + G(v) ) ]^z
	//using symbolic mathematics

	//Gaussian distributions
	auto Gx = hydra::Gaussian<_x>(mean_x, sigma);
	auto Gy = hydra::Gaussian<_y>(mean_y, sigma);
	auto Gu = hydra::Gaussian<_u>(mean_u, sigma);
	auto Gv = hydra::Gaussian<_v>(mean_v, sigma);

	auto Axy =  (Gx - Gy)/(Gx + Gy);
	auto Auv =  (Gu - Gv)/(Gu + Gv);
    auto A   =  Axy + Auv;

	auto powerz = hydra::wrap_lambda( [] __hydra_dual__ (unsigned int npar, const hydra::Parameter* params, double x )
	{

		return ::pow(x, params[0]);

	}, exponent );

	auto Total = hydra::compose(powerz, A );

	//print parameters
	Total.PrintRegisteredParameters();

	//evaluate using named function arguments
    std::cout << Total( _x(1.0), _y(-1.0), _v(1.0), _u(-1.0)) << std::endl;
    //evaluate using named function arguments (changed order)
    std::cout << Total( _y(-1.0), _x(1.0), _u(-1.0), _v(1.0)) << std::endl;
    //evaluate using tuple of unamed arguments (risky!!!)
    std::cout << Total( hydra::make_tuple(1.0, -1.0, 1.0, -1.0)) << std::endl;
    //evaluate using tuple of unamed arguments (risky!!!)
    std::cout << Total( hydra::make_tuple(-1.0, 1.0, -1.0, 1.0)) << std::endl;
    //evaluate using tuple of unamed arguments (risky!!!)
    std::cout << Total( hydra::make_tuple(1.0, -1.0, -1.0, 1.0)) << std::endl;
    //evaluate using tuple of named arguments ()
    std::cout << Total( hydra::make_tuple( _x(1.0), _y(-1.0), _v(1.0), _u(-1.0))) << std::endl;


    auto myfunc = hydra::wrap_lambda( [] __hydra_dual__ (unsigned int npar, const hydra::Parameter* params, _x x, _y y) {

    	return x + params[0]*y;

    }, exponent);

    auto mycombination = Gx + myfunc * Gy;

    std::cout << "mycombination: " << mycombination(_x(-1.0), _y(1.0)) << std::endl;

    //What is the type of the natural signature of Total?
    //uncomment this
   // typename decltype(mycombination)::argument_type  test{};
    //std::cout << test.dummy << '\n';

	return 0;
}



#endif /* QUICK_TEST_INL_ */
