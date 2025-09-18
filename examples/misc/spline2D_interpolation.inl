/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2025 Antonio Augusto Alves Junior
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
 * spline2D_interpolation.inl
 *
 *  Created on: 17/09/2023
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPILINE2D_INTERPOLATION_INL_
#define SPILINE2D_INTERPOLATION_INL_


#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>

//command line
#include <tclap/CmdLine.h>

//this lib
#include <hydra/device/System.h>
#include <hydra/Function.h>
#include <hydra/Lambda.h>
#include <hydra/functions/Gaussian.h>
#include <hydra/functions/UniformShape.h>
#include <hydra/functions/Spline2DFunctor.h>
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
	hydra::Parameter  sigma = hydra::Parameter::Create().Name("Sigma").Value(3.0).Error(0.0001).Limits(0.01, 1.5);


	//gaussian function evaluating on argument zero
	hydra::Gaussian<double> gaussian(mean, sigma);


    //set the x dimension of the grid
	auto xaxis =  hydra::range(-10.0, 10.0, 50);
	auto x_grid_size = xaxis.size();
	auto xiter = xaxis.begin();

	//set the y dimension of the grid
	auto yaxis =  hydra::range(-10.0, 10.0, 40);
	auto y_grid_size = yaxis.size();
        auto yiter = yaxis.begin();

	auto gaussian_2D = hydra::wrap_lambda(
			[gaussian, xiter,x_grid_size, yiter, y_grid_size ] __hydra_dual__ ( size_t index){

		unsigned j = index/x_grid_size ;
		unsigned i = index%x_grid_size ;
        auto x = xiter[i];
        auto y = yiter[j];
        auto r = gaussian( x )*gaussian( y );

		return r;
	});

	auto index = hydra::range(0, x_grid_size*y_grid_size);

	auto ordinate  = index | gaussian_2D;

	auto spline2D = hydra::make_spline2D<double, double>(xaxis, yaxis, ordinate );

	//get random values for x and y
	auto random_x = hydra::random_range( hydra::UniformShape<double>(-10.0, 10.0), 157531, 10) ;
	auto random_y = hydra::random_range( hydra::UniformShape<double>(-10.0, 10.0), 456258, 10) ;


	for( auto x:random_x ){
		for( auto y:random_y ){

			printf(" x %f y %f spline2D %f gaussian2D %f\n", x,y,
					spline2D(x,y), gaussian(x)*gaussian( y ));
		}
	}



return 0;
}


#endif /* SPILINE_INTERPOLATION_INL_ */
