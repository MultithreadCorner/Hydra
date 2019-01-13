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
#include <vector>

//hydra
#include <hydra/FFTW.h>
#include <hydra/device/System.h>
#include <hydra/Algorithm.h>
#include <hydra/Random.h>
#include <hydra/Zip.h>
#include <hydra/Complex.h>
#include <hydra/detail/utility/Utility_Tuple.h>

//command line
#include <tclap/CmdLine.h>



typedef double FloatType;

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
	 * Real -> Complex -> Real
	 */
	hydra::device::vector<FloatType> x( nentries);

	//generate random
	hydra::Random<> Generator{};
	Generator.Uniform(-1.0, 1.0, x);


   auto fft_r2c = hydra::RealToComplexFFTW<FloatType>( nentries );
   fft_r2c.LoadInputData(x);
   fft_r2c.Execute();

   auto r2c_out =  fft_r2c.GetOutputData();

   auto output_r2c = hydra::make_range(r2c_out.first,
		   r2c_out.first + r2c_out.second );

   auto fft_c2r = hydra::ComplexToRealFFTW<FloatType>( nentries );

   fft_c2r.LoadInputData( r2c_out.second, r2c_out.first );

   fft_c2r.Execute();

   auto c2r_out =  fft_c2r.GetOutputData();

   auto output_c2r = hydra::make_range( c2r_out.first,
		   c2r_out.first + c2r_out.second);


   auto data = hydra::zip( x, output_r2c, output_c2r );

   printf(" ---- real ---- | ---------- complex ---------- | ----- real -----\n");

   hydra::for_each( data ,
		   [nentries] __hydra_dual__ ( hydra::tuple<FloatType, hydra::complex<FloatType>, FloatType>  a){

	   printf("%f \t| %f:re + %f:im \t| %f \n", hydra::get<0>(a),
			   hydra::get<1>(a).real(), hydra::get<1>(a).imag(), hydra::get<2>(a)/nentries );
	  });

   //---------------------------------------------------------------------

   hydra::device::vector<hydra::complex<FloatType>> c(nentries,
		     hydra::complex<FloatType>(1.0,2.0) );

   auto fft_c2c_f = hydra::ComplexToComplexFFTW<FloatType>( nentries , +1);

   fft_c2c_f.LoadInputData( c );

   fft_c2c_f.Execute();

   auto c2c_out_f =  fft_c2c_f.GetOutputData();

   auto output_c2c_f = hydra::make_range( c2c_out_f.first,
		   c2c_out_f.first + c2c_out_f.second);

   auto fft_c2c_b = hydra::ComplexToComplexFFTW<FloatType>( nentries , -1);

    fft_c2c_b.LoadInputData( c2c_out_f.second,  c2c_out_f.first);

    fft_c2c_b.Execute();

    auto c2c_out_b =  fft_c2c_b.GetOutputData();

    auto output_c2c_b = hydra::make_range( c2c_out_b.first,
 		   c2c_out_b.first + c2c_out_b.second);



   auto datac = hydra::zip( c, output_c2c_f , output_c2c_b);

   printf(" ----------- complex ---------- | ---------- complex ---------- | ---------- complex ----------\n");
   hydra::for_each(datac , [nentries] __hydra_dual__ ( hydra::tuple< hydra::complex<FloatType>, hydra::complex<FloatType>, hydra::complex<FloatType>>  a)
   {

	   printf(" %f:re + %f:im \t| %f:re + %f:im \t| %f:re + %f:im \n",
			   hydra::get<0>(a).real(), hydra::get<0>(a).imag(),
			   hydra::get<1>(a).real()/nentries, hydra::get<1>(a).imag()/nentries,
			   hydra::get<2>(a).real()/nentries, hydra::get<2>(a).imag()/nentries );
	  });


	return 0;
}

#endif /* FFT_INL_ */
