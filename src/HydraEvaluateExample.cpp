/*
 * TestHydra.cu
 *
 *  Created on: Jun 21, 2016
 *      Author: augalves
 */



#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <time.h>
#include <string>
#include <vector>
#include <array>
#include <tuple>
#include <chrono>
#include <type_traits>
#include <typeinfo>
//command line
#include <tclap/CmdLine.h>
#define CUDA_API_PER_THREAD_DEFAULT_STREAM

//this lib
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/Evaluate.h>
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Random.h>


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

using namespace std;
using namespace hydra;


GInt_t main(int argv, char** argc)
{
	typedef std::pair<GUInt_t, GReal_t> Var_t;

	size_t nentries = 0;

	try {

		TCLAP::CmdLine cmd("Command line arguments for HydraRandomExample", '=');

		TCLAP::ValueArg<GULong_t> EArg("n", "number-of-events",
				"Number of events",
				true, 5e6, "long");
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
	//Print::SetLevel(0);
	//ROOT::Minuit2::MnPrint::SetLevel(2);
	//----------------------------------------------

	//Generator with current time count as seed.
	size_t seed = std::chrono::system_clock::now().time_since_epoch().count();
	Random<> Generator( seed  );


	//-------------------------------------------
	//range of the analysis
	std::array<GReal_t, 1>  min   = { 0.0};
	std::array<GReal_t, 1>  max   = { 2*PI};

	//Generate data
	RealVector_d angles_d(nentries);
	Generator.Uniform(min[0] , max[0], angles_d.begin(), angles_d.end() );


	auto sin_lambda = [] __host__ __device__(GReal_t* x)
	{ return sin(x[0]); };

	auto sin_lambaW  = LambdaWrapper<GReal_t(GReal_t* x),
			decltype(sin_lambda) >(sin_lambda);

	auto cos_lambda = [] __host__ __device__(GReal_t*  x)
	{ return cos(x[0]); };

	auto cos_lambaW  = LambdaWrapper<GReal_t(GReal_t* x),
			decltype(cos_lambda) >(cos_lambda);



	auto R2_lambda = [] __host__ __device__(thrust::tuple<GReal_t, GReal_t>* x)
	{ return thrust::get<0>(*x)*thrust::get<0>(*x) +  thrust::get<1>(*x)*thrust::get<1>(*x); };

	auto R2_lambdaW  = wrap_lambda(R2_lambda);

			//LambdaWrapper<GReal_t(thrust::tuple<GReal_t, GReal_t>* x),
			//decltype(R2_lambda) >(R2_lambda);

   // static_assert(decltype(&decltype(R2_lambda)::operator())::er, "mydummy");

	auto functors = thrust::make_tuple( sin_lambaW, cos_lambaW);
	auto range =make_range( angles_d.begin(), angles_d.end());



	auto result = Eval( functors, range);

    auto range2 =make_range( result.begin(), result.end());
	auto result2 = Eval( R2_lambdaW, range2);



	for(size_t i=0; i< 10; i++ )
		std::cout << i <<" [sin(x), cos(x)] = " << result[i]
		               <<"\t [sin(x)^2 + cos(x)^2] = " << result2[i] << std::endl;
	return 0;

}
