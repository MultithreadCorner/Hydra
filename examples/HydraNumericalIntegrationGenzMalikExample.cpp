/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 Antonio Augusto Alves Junior
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
 * HydraFitExample.cu
 *
 *  Created on: Jun 21, 2016
 *      Author: Antonio Augusto Alves Junior
 */

#include <iostream>
#include <assert.h>
#include <time.h>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <limits>
//command line arguments
#include <tclap/CmdLine.h>

//this lib
#include <hydra/Types.h>
#include <hydra/Vector4R.h>
#include <hydra/Containers.h>
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/Parameter.h>
#include <hydra/experimental/GenzMalikQuadrature.h>
#include <hydra/detail/utility/Permute.h>
//root
#include <TROOT.h>
#include <TH1D.h>
#include <TF1.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TString.h>
#include <TStyle.h>

#include <examples/Gauss.h>
#include <examples/Exp.h>

using namespace std;
using namespace hydra;
using namespace examples;

namespace examples {

struct Unit
{
	template<typename T>
	 double operator()(T x){return 1;}

};


}  // namespace examples


GInt_t main(int argv, char** argc)
{

	size_t nboxes              = 0;

	try {

		TCLAP::CmdLine cmd("Command line arguments for HydraNumericalIntegrationGenzMalik", '=');

		TCLAP::ValueArg<size_t> NBoxesArg("n", "number-of-boxes",
				"Number of boxes",
				true, 10, "size_t");
		cmd.add( NBoxesArg);

		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		nboxes     = NBoxesArg.getValue();

	}
	catch (TCLAP::ArgException &e)
	{
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}


	constexpr size_t N = 10;

	//------------------------------------
	//parameters
	//------------------------------------

	std::string  Mean_s[N];
	std::string Sigma_s[N];
	GUInt_t  Position_p[N];
	Parameter    Mean_p[N];
	Parameter   Sigma_p[N];

    //-------------------------------------------
	//range of the analysis
	std::array<GReal_t, N>  min;
	std::array<GReal_t, N>  max;

	for(size_t i=0; i< N; i++){

		    min[i] = -6.0;
		    max[i] =  6.0;
	 Position_p[i] = i;
		 Mean_s[i] = "mean_"  ;
		 Mean_s[i] += std::to_string(i);
		Sigma_s[i] = "sigma_" ;
		Sigma_s[i] += std::to_string(i);
		 Mean_p[i].Name(Mean_s[i]).Value(0.0) .Error(0.0001).Limits( -5.0, 5.0);
		Sigma_p[i].Name(Sigma_s[i]).Value(1.0) .Error(0.0001).Limits( 0.5, 1.5);
	}

	//----------------------------------------------------------------------
	// create functor
	//------------------------------------

	GaussN<N> Gaussian(Mean_p, Sigma_p, Position_p);



	//----------------------------------------------------------------------
	//ANALYTIC
	//----------------------------------------------------------------------
	GaussNAnalyticIntegral<N> gaussianAnaInt(min, max);
	auto result = gaussianAnaInt.Integrate(Gaussian);

	cout << ">>> Gaussian intetgral [Analytic]"<< endl;
	cout << "Result: " << std::setprecision(10)<<result.first
					   << " +/- "    << result.second <<std::endl;


	//----------------------------------------------------------------------
	//Genz-Malik
	//----------------------------------------------------------------------

	auto GMIntegrator = hydra::experimental::GenzMalikQuadrature<N, hydra::device>(min, max, nboxes);
	auto start = std::chrono::high_resolution_clock::now();
	auto result2 = GMIntegrator.Integrate(Gaussian);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed = end - start;


	cout << ">>> Gaussian intetgral [Genz-Malik]"<< endl;
	cout << "Result: " << std::setprecision(10)<<result2.first
					   << " +/- "    << result2.second <<std::endl <<
					   " Time (ms): "<< elapsed.count() <<std::endl;


    /*
	int A[5]{0,1,2,3,4};
    hydra::detail::nth_permutation(A, A+5, 1, thrust::less<int>() );
    */
	return 0;


	}

