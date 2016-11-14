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
 * HydraEvaluateExample.cu
 *
 *  Created on: Jun 21, 2016
 *      Author: Antonio Augusto Alves Junior
 */


#include <iostream>
#include <string>
#include <array>
#include <chrono>
#include <time.h>
//command line
#include <tclap/CmdLine.h>

//this lib
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/Evaluate.h>
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Random.h>



using namespace std;
using namespace hydra;

/**
 * @file
 * @example HydraEvaluateExample.cu
 * @brief This is an example of how to use hydra::Eval to evaluate C++11 lambdas
 * using the CUDA backend.
 * The usage and the expected output is something like this:
 ```
 ./Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Eval -n=10000000

--------------------------------------------------------------
| Evaluation of [sin(x), cos(x)]
| Time (ms) = 0.268075
--------------------------------------------------------------
--------------------------------------------------------------
| Evaluation of [sin(x)^2 + cos(x)^2]
| Time (ms) = 0.213697
--------------------------------------------------------------
|>   0 [sin(x), cos(x)] = (-0.252592278 -0.967572809) ............... [sin(x)^2 + cos(x)^2] = 1
|>   1 [sin(x), cos(x)] = (0.932195527 -0.361955107) ............... [sin(x)^2 + cos(x)^2] = 1
|>   2 [sin(x), cos(x)] = (0.585050502 0.810996862) ............... [sin(x)^2 + cos(x)^2] = 1
|>   3 [sin(x), cos(x)] = (0.851355528 0.524589139) ............... [sin(x)^2 + cos(x)^2] = 1
|>   4 [sin(x), cos(x)] = (0.921620144 0.388093171) ............... [sin(x)^2 + cos(x)^2] = 1
|>   5 [sin(x), cos(x)] = (-0.989926331 0.141583398) ............... [sin(x)^2 + cos(x)^2] = 1
|>   6 [sin(x), cos(x)] = (-0.775938693 -0.630808327) ............... [sin(x)^2 + cos(x)^2] = 1
|>   7 [sin(x), cos(x)] = (0.879555884 0.475795593) ............... [sin(x)^2 + cos(x)^2] = 1
|>   8 [sin(x), cos(x)] = (0.964678017 0.263431819) ............... [sin(x)^2 + cos(x)^2] = 1
|>   9 [sin(x), cos(x)] = (-0.999776981 -0.0211184451) ............... [sin(x)^2 + cos(x)^2] = 1

 ```
*/



/**
 * @file
 * @brief This is an example of how to use hydra::Eval to evaluate C++11 lambdas
 * using the CUDA backend.
 * The usage and the expected output is something like this:
 ```
 ./Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Eval -n=10000000

--------------------------------------------------------------
| Evaluation of [sin(x), cos(x)]
| Time (ms) = 0.268075
--------------------------------------------------------------
--------------------------------------------------------------
| Evaluation of [sin(x)^2 + cos(x)^2]
| Time (ms) = 0.213697
--------------------------------------------------------------
|>   0 [sin(x), cos(x)] = (-0.252592278 -0.967572809) ............... [sin(x)^2 + cos(x)^2] = 1
|>   1 [sin(x), cos(x)] = (0.932195527 -0.361955107) ............... [sin(x)^2 + cos(x)^2] = 1
|>   2 [sin(x), cos(x)] = (0.585050502 0.810996862) ............... [sin(x)^2 + cos(x)^2] = 1
|>   3 [sin(x), cos(x)] = (0.851355528 0.524589139) ............... [sin(x)^2 + cos(x)^2] = 1
|>   4 [sin(x), cos(x)] = (0.921620144 0.388093171) ............... [sin(x)^2 + cos(x)^2] = 1
|>   5 [sin(x), cos(x)] = (-0.989926331 0.141583398) ............... [sin(x)^2 + cos(x)^2] = 1
|>   6 [sin(x), cos(x)] = (-0.775938693 -0.630808327) ............... [sin(x)^2 + cos(x)^2] = 1
|>   7 [sin(x), cos(x)] = (0.879555884 0.475795593) ............... [sin(x)^2 + cos(x)^2] = 1
|>   8 [sin(x), cos(x)] = (0.964678017 0.263431819) ............... [sin(x)^2 + cos(x)^2] = 1
|>   9 [sin(x), cos(x)] = (-0.999776981 -0.0211184451) ............... [sin(x)^2 + cos(x)^2] = 1

 ```
*/
GInt_t main(int argv, char** argc)
{

	/// number of entries or trials
	size_t nentries = 0;

	//use tclap to get the command line parameters
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

	//Generator with current time count as seed.
	size_t seed = std::chrono::system_clock::now().time_since_epoch().count();

	//hydra::Random with default template parameters
	Random<> Generator( seed );

	//1D range
	std::array<GReal_t, 1>  min = {0.0};
	std::array<GReal_t, 1>  max = {2.0*PI};

	//allocate a vector of real numbers in the device
	RealVector_d angles_d(nentries);

	//Generate angles uniformly in the range
	//Generation will be performed in the device
	Generator.Uniform(min[0] , max[0], angles_d.begin(), angles_d.end() );

	//Simple c++11 lambda function to calculate the sin of angles
	auto sin_lambda = [] __host__ __device__(GReal_t* x)
	{ return sin(x[0]); };

	//Wrap the lambda
	auto sin_lambaW  = wrap_lambda(sin_lambda);

	//Simple c++11 lambda function to calculate the cos of angles
	auto cos_lambda = [] __host__ __device__(GReal_t*  x)
	{ return cos(x[0]); };

	//Wrap the lambda
	auto cos_lambaW  = wrap_lambda(cos_lambda);

	//Simple c++11 lambda function to calculate the {cos(angle)}^2 + {sin(angle)}^2
	auto R2_lambda = [] __host__ __device__(thrust::tuple<GReal_t, GReal_t>* x)
	{ return thrust::get<0>(*x)*thrust::get<0>(*x) +  thrust::get<1>(*x)*thrust::get<1>(*x); };

	//Wrap the lambda
	auto R2_lambdaW  = wrap_lambda(R2_lambda);

	//Aggregate the functors in a tuple
   	auto functors = thrust::make_tuple( sin_lambaW, cos_lambaW);

   	//Define a range
	auto range    = make_range( angles_d.begin(), angles_d.end());

    //start time
	auto start1 = std::chrono::high_resolution_clock::now();
	//Evaluate sin and cos
	auto result = Eval( functors, range);
	//end time
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
	//time
	std::cout << "--------------------------------------------------------------"<<std::endl;
	std::cout << "| Evaluation of [sin(x), cos(x)] "<<std::endl;
	std::cout << "| Time (ms) = "<< elapsed1.count() <<std::endl;
	std::cout << "--------------------------------------------------------------"<<std::endl;

	//Define a second range
    auto range2 =make_range( result.begin(), result.end());

    //start time
    auto start2 = std::chrono::high_resolution_clock::now();
    //Evaluate {cos(angle)}^2 + {sin(angle)}^2
    auto result2 = Eval( R2_lambdaW, range2);
    //end time
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed2 = end2 - start2;
    //time
    std::cout << "--------------------------------------------------------------"<<std::endl;
    std::cout << "| Evaluation of [sin(x)^2 + cos(x)^2] "<<std::endl;
    std::cout << "| Time (ms) = "<< elapsed2.count() <<std::endl;
    std::cout << "--------------------------------------------------------------"<<std::endl;

	//Print output
	for(size_t i=0; i< 10; i++ )
		std::cout<<"|> " << std::setfill (' ') << std::setw(3)  << i <<" [sin(x), cos(x)] = " << std::setprecision(9) << result[i]
		<<" " << std::setfill ('.') << std::setw (40)<<" [sin(x)^2 + cos(x)^2] = " << std::setprecision(9) << result2[i] << std::endl;


return 0;



}


