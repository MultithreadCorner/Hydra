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
 * fft.inl
 *
 *  Created on: 14/11/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FFT_INL_
#define FFT_INL_

#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>

//hydra
#include <hydra/FFTCPU.h>
#include <hydra/device/System.h>
#include <hydra/Algorithm.h>
#include <hydra/Random.h>
#include <hydra/Zip.h>
//command line
#include <tclap/CmdLine.h>


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

	hydra::device::vector<float> x( nentries);


	//generate random
	hydra::Random<> Generator{};
	Generator.Uniform(-1.0, 1.0, x);


   auto fft_r2c = hydra::RealToComplexFFT<float>( nentries );
   fft_r2c.LoadInputData(x);
   fft_r2c.Execute();

   auto output_r2c = hydra::make_range(fft_r2c.GetTransformedData().first,
		   fft_r2c.GetTransformedData().first + fft_r2c.GetTransformedData().second );

   auto fft_c2r = hydra::ComplexToRealFFT<float>( nentries );

   fft_c2r.LoadInputData(fft_r2c.GetTransformedData().second,
		   fft_r2c.GetTransformedData().first );

   fft_c2r.Execute();

   auto output_c2r = hydra::make_range(fft_c2r.GetTransformedData().first,
		   fft_c2r.GetTransformedData().first + fft_c2r.GetTransformedData().second );

   auto data = hydra::zip( x, output_r2c, output_c2r );
   hydra::for_each( data ,
		   [nentries] __hydra_dual__ ( hydra::tuple<double,double*, double> a){

	   printf("%f | %f:re + %f:im | %f \n", hydra::get<0>(a),  hydra::get<1>(a)[0], hydra::get<1>(a)[1], hydra::get<2>(a)/nentries );
   });



	return 0;
}

#endif /* FFT_INL_ */
