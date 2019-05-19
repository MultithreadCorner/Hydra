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
 * variant_types.inl
 *
 *  Created on: 22/10/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef VARIANT_TYPES_INL_
#define VARIANT_TYPES_INL_

#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>

#include <hydra/device/System.h>
#include <hydra/host/System.h>

#include <hydra/Vector3R.h>
#include <hydra/Vector4R.h>
#include <hydra/Complex.h>
#include <hydra/Tuple.h>
#include <hydra/Variant.h>

//command line
#include <tclap/CmdLine.h>

struct setter_t{

		setter_t(double x, double y, double z, double w):	components{x,y,z,w}{}
		setter_t(double x, double y, double z):components{x,y,z, 0.0}{}
		setter_t(double x, double y):components{x,y, 0.0, 0.0}{}

		void operator()( hydra::complex<double>& v){v.real( components[0] ); v.imag( components[1] ); }
		void operator()( hydra::Vector3R& v){v.set( components[0], components[1], components[2]);}
		void operator()( hydra::Vector4R& v){v.set(components[0], components[1], components[2], components[3]);}

		double components[4];
	};

	struct printer_t{

		void operator()( hydra::complex<double>& v){std::cout <<"Complex :"<< v << std::endl;}

		void operator()( hydra::Vector3R& v){std::cout <<"Vector3R :"<< v << std::endl;}

		void operator()( hydra::Vector4R& v){std::cout <<"Vector4R :"<< v << std::endl;}

	};


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


	/*
	 * storing different types in vector
	 */
	typedef hydra::experimental::variant< hydra::complex<double>,hydra::Vector3R,  hydra::Vector4R> composite_t;

	composite_t composite;

	composite = hydra::complex<double>(1.0,2.0);
	hydra::experimental::visit(printer_t(), composite);
	hydra::experimental::visit(setter_t(2.0, 4.0), composite);
	hydra::experimental::visit(printer_t(), composite);

	composite = hydra::Vector3R(1.0,2.0,3.0);
	hydra::experimental::visit(printer_t(), composite);
	hydra::experimental::visit(setter_t(2.0, 4.0, 6.0), composite);
	hydra::experimental::visit(printer_t(), composite);

	composite = hydra::Vector4R(1.0,2.0,3.0,4.0);
	hydra::experimental::visit(printer_t(), composite);
	hydra::experimental::visit(setter_t(2.0, 4.0, 6.0,8.0), composite);
	hydra::experimental::visit(printer_t(), composite);


	/**
	 * vector
	 */
	typedef hydra::experimental::variant< hydra::Vector3R > number_t;
	hydra::experimental::get<0>(number_t());

	hydra::device::vector<composite_t > Vector;
	for(size_t i=0; i<10; i++)
		Vector.push_back(hydra::complex<double>(1.0,2.0));
	for(size_t i=0; i<10; i++)
			Vector.push_back(hydra::Vector3R(1.0,2.0,3.0));
	for(size_t i=0; i<10; i++)
				Vector.push_back(hydra::Vector4R(1.0,2.0,3.0,4.0));

	for(composite_t el:Vector)
	       hydra::experimental::visit(printer_t(),el);
	std::cout<< "===============>" << std::endl << std::endl;

	for(auto el:Vector)
		el = hydra::Vector3R(2.0,4.0,6.0);

	for(composite_t el:Vector)
		       hydra::experimental::visit(printer_t(),el);
	return 0;
}

#endif /* VARIANT_TYPES_INL_ */
