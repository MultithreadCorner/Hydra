/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
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
 * spline4D_interpolation.inl
 *
 *  Created on: 17/09/2023
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPILINE4D_INTERPOLATION_INL_
#define SPILINE4D_INTERPOLATION_INL_


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
#include <hydra/functions/Spline4DFunctor.h>
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
	hydra::Gaussian<double> gaussian(mean, sigma);

	//set the x dimension of the grid
	auto xaxis =  hydra::range(-3.0, 3.0, 200);
	auto x_grid_size = xaxis.size();
	auto xiter = xaxis.begin();

	//set the y dimension of the grid
	auto yaxis =  hydra::range(-3.0, 3.0, 200);
	auto y_grid_size = yaxis.size();
	auto yiter = yaxis.begin();

	//set the w dimension of the grid
	auto waxis =  hydra::range(-3.0, 3.0, 200);
	auto w_grid_size = waxis.size();
	auto witer = waxis.begin();

	//set the z dimension of the grid
	auto zaxis =  hydra::range(-3.0, 3.0, 200);
	auto z_grid_size = zaxis.size();
	auto ziter = zaxis.begin();



		auto gaussian_4D = hydra::wrap_lambda(
				[gaussian, xiter,x_grid_size, yiter, y_grid_size, witer, w_grid_size , ziter, z_grid_size ] __hydra_dual__ ( size_t index){

			unsigned ix = index%x_grid_size ;
			unsigned iy = (index/x_grid_size)%y_grid_size ;
			unsigned iw = (index/(x_grid_size*y_grid_size)) %w_grid_size ;
			unsigned iz = index/(x_grid_size*y_grid_size*w_grid_size) ;


	        auto x = xiter[ix];
	        auto y = yiter[iy];
	        auto w = witer[iw];
	        auto z = ziter[iz];

	        auto r =gaussian( x )*gaussian( y )*gaussian( w )*gaussian( z );

			return r;
		});

		auto index = hydra::range(0, x_grid_size*y_grid_size*w_grid_size*z_grid_size);

		auto ordinate  = index | gaussian_4D;

		auto spline4D = hydra::make_spline4D<double, double, double, double>(xaxis, yaxis, waxis, zaxis, ordinate );

		//get random values for x and y
		auto random_x = hydra::random_range( hydra::UniformShape<double>(-1.0, 1.0), 157531, 10) ;
		auto random_y = hydra::random_range( hydra::UniformShape<double>(-1.0, 1.0), 456258, 10) ;
		auto random_w = hydra::random_range( hydra::UniformShape<double>(-1.0, 1.0), 753159, 10) ;
		auto random_z = hydra::random_range( hydra::UniformShape<double>(-1.0, 1.0), 789512, 10) ;



		for( auto x:random_x ){
			for( auto y:random_y ){
				for( auto w:random_w ){
					for( auto z:random_z ){
						printf(" x %f y %f w %f z %f spline4D %f gaussian4D %f\n", x,y, w, z,
								spline4D(x,y, w, z), gaussian(x)*gaussian( y )*gaussian( w )*gaussian( z ));
					}
				}
			}
		}


return 0;
}


#endif /* SPILINE_INTERPOLATION_INL_ */
