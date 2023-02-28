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
#include <hydra/Lambda.h>
#include <hydra/Tuple.h>
#include <hydra/multivector.h>
#include <hydra/Placeholders.h>
#include <hydra/Random.h>
#include <hydra/Algorithm.h>
#include <hydra/DenseHistogram.h>
#include <hydra/Zip.h>
#include <hydra/functions/UniformShape.h>

//command line arguments
#include <tclap/CmdLine.h>


#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"

declarg(AxisX, double)
declarg(AxisY, double)
declarg(AxisZ, double)
declarg(Rho  , double)


using namespace hydra::arguments;
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
	auto length = hydra::wrap_lambda([] __hydra_dual__ (AxisX x, AxisY y) {

		return Rho(::sqrt( x*x + y*y));
	});

	// flag according distance to origin
	auto is_inside =  hydra::wrap_lambda(
			[] __hydra_dual__ (Rho r){

		return   r > 1.0 ;
	});

	//device
	{
		std::cout << "=========================================="<<std::endl;
		std::cout << "|            <--- DEVICE --->            |"<<std::endl;
		std::cout << "=========================================="<<std::endl;


		hydra::multivector<hydra::tuple<AxisX, AxisY, AxisZ>, hydra::device::sys_t> positions(
				hydra::zip(
				hydra::random_range( hydra::UniformShape<AxisX>(-1.5, 1.5), 753, nentries),
				hydra::random_range( hydra::UniformShape<AxisY>(-1.5, 1.5), 123, nentries),
				hydra::random_range( hydra::UniformShape<AxisZ>(-1.5, 1.5), 789, nentries))
		 );
		std::cout  << std::endl<< std::endl<< std::endl;
		std::cout  << std::endl<<"hydra::sort_by_key" <<std::endl<< std::endl;
		for(auto i : positions) {
			if(is_inside(length(i))) std::cout << "\033[1;31m" << i << ", ";
			else  std::cout << "\033[1;33m" << i << ", ";
		}  std::cout <<"\033[0m\n";

		auto sorted_range = hydra::sort_by_key(positions, hydra::columns(positions, _0,_1 ) | length | is_inside )  ;

		std::cout  << std::endl<< std::endl<< std::endl;

		for(auto i : positions) {
			if( is_inside(length(i)) ) std::cout << "\033[1;31m" << i << ", ";
			else  std::cout << "\033[1;33m" << i << ", ";
		}  std::cout <<"\033[0m\n";

		std::cout  << std::endl<< std::endl<< std::endl;
		std::cout  << std::endl<<"hydra::for_each" <<std::endl<< std::endl;
		hydra::for_each(positions, hydra::wrap_lambda(
				[ is_inside, length ] __hydra_dual__ (AxisX x, AxisY y, AxisZ z ){

			if( !is_inside(length(x, y)) ){
				printf(ANSI_COLOR_RED "Inside: %f %f %f\n " ANSI_COLOR_RESET, x(), y(), z());
			}
			else {
				printf(ANSI_COLOR_GREEN "Outside %f %f %f\n " ANSI_COLOR_RESET, x(), y(), z());
			}

		}));


		std::array<double, 3> masses{0.13957061, 0.13957061,0.13957061};

		auto events =  hydra::phase_space_range(hydra::Vector4R(0.493677, 0.0, 0.0, 0.0), masses, 321,nentries );

		auto invariant_mass = hydra::wrap_lambda(
				[]__hydra_dual__( double Weights, hydra::Vector4R A, hydra::Vector4R B, hydra::Vector4R C){

			return (A + B).mass();
		});

		hydra::DenseHistogram<double,1, hydra::device::sys_t> Hist_Mass(10, masses[0]+masses[1], 0.493677 - masses[0]);

		hydra::for_each( Hist_Mass.Fill( events | invariant_mass ),
				 hydra::wrap_lambda( [] __hydra_dual__ ( double a){ printf("%f, ", a); } ) );


	}//device


	return 0;
}



#endif /* RANGE_SEMANTICS_INL_ */
