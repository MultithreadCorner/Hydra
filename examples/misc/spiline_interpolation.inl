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
 * spiline_interpolation.inl
 *
 *  Created on: 23/12/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPILINE_INTERPOLATION_INL_
#define SPILINE_INTERPOLATION_INL_


#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>

//command line
#include <tclap/CmdLine.h>

//this lib
#include <hydra/device/System.h>
#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/functions/Gaussian.h>
#include <hydra/functions/SpilineFunctor.h>
#include <hydra/Range.h>
#include <hydra/Algorithm.h>
/*-------------------------------------
 * Include classes from ROOT to fill
 * and draw histograms and plots.
 *-------------------------------------
 */
#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <TH1D.h>
#include <TApplication.h>
#include <TCanvas.h>

#endif //_ROOT_AVAILABLE_

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

	//parameters
	hydra::Parameter  mean  = hydra::Parameter::Create().Name("Mean").Value(0.0).Error(0.0001).Limits(-1.0, 1.0);
	hydra::Parameter  sigma = hydra::Parameter::Create().Name("Sigma").Value(1.0).Error(0.0001).Limits(0.01, 1.5);


	//gaussian function evaluating on argument zero
	hydra::Gaussian<> gaussian(mean, sigma);

	auto abscissae = hydra::device::vector<double>(21);
	hydra::copy( hydra::range(-10, 10), abscissae );

	auto ordinate  = abscissae | gaussian;

	auto spiline = hydra::make_spiline(abscissae, ordinate );

    hydra::device::vector<double> args(50, 1.0);


    std::cout <<  "Size = " <<  hydra::random_uniform_range(-10.0, 10.0, 15753).size() << std::endl;

    hydra::for_each(args , [ spiline]__hydra_dual__(double arg){
    	printf("arg %f spiline %f\n", arg, spiline(arg));
    } );



return 0;
}


#endif /* SPILINE_INTERPOLATION_INL_ */
