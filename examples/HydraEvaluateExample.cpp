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
 * HydraEvaluateExample.cpp
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

#include <hydra/host/System.h>
#include <hydra/omp/System.h>
#include <hydra/cpp/System.h>

using namespace std;
using namespace hydra;

/**
 * @file
 * @example HydraEvaluateExample.cpp
 * @brief This is an example of how to use hydra::Eval to evaluate C++11 lambdas using the OpenMP backend.
 * The usage and the expected output is something like this:
```
./Hydra_Example_GCC_DEVICE_OMP_HOST_CPP_Eval -n=10000000

--------------------------------------------------------------
| Evaluation of [sin(x), cos(x)]
| Time (ms) = 534.884
--------------------------------------------------------------
--------------------------------------------------------------
| Evaluation of [sin(x)^2 + cos(x)^2]
| Time (ms) = 21.6937
--------------------------------------------------------------
|>   0 [sin(x), cos(x)] = (-0.303346204 -0.952880412) ............... [sin(x)^2 + cos(x)^2] = 1
|>   1 [sin(x), cos(x)] = (0.974836209 -0.222922333) ............... [sin(x)^2 + cos(x)^2] = 1
|>   2 [sin(x), cos(x)] = (-0.69576933 0.718265299) ............... [sin(x)^2 + cos(x)^2] = 1
|>   3 [sin(x), cos(x)] = (0.853703285 -0.520759734) ............... [sin(x)^2 + cos(x)^2] = 1
|>   4 [sin(x), cos(x)] = (-0.941210424 0.337820866) ............... [sin(x)^2 + cos(x)^2] = 1
|>   5 [sin(x), cos(x)] = (-0.8711111 -0.491085992) ............... [sin(x)^2 + cos(x)^2] = 1
|>   6 [sin(x), cos(x)] = (-0.0704802093 -0.997513178) ............... [sin(x)^2 + cos(x)^2] = 1
|>   7 [sin(x), cos(x)] = (0.490961891 -0.87118105) ............... [sin(x)^2 + cos(x)^2] = 1
|>   8 [sin(x), cos(x)] = (-0.78756139 0.616236202) ............... [sin(x)^2 + cos(x)^2] = 1
|>   9 [sin(x), cos(x)] = (0.715995247 0.698105154) ............... [sin(x)^2 + cos(x)^2] = 1

```
 */



/**
 * @file
 * @brief This is an example of how to use hydra::Eval to evaluate C++11 lambdas using the OpenMP backend.
 * The usage and the expected output is something like this:
```
./Hydra_Example_GCC_DEVICE_OMP_HOST_CPP_Eval -n=10000000

--------------------------------------------------------------
| Evaluation of [sin(x), cos(x)]
| Time (ms) = 534.884
--------------------------------------------------------------
--------------------------------------------------------------
| Evaluation of [sin(x)^2 + cos(x)^2]
| Time (ms) = 21.6937
--------------------------------------------------------------
|>   0 [sin(x), cos(x)] = (-0.303346204 -0.952880412) ............... [sin(x)^2 + cos(x)^2] = 1
|>   1 [sin(x), cos(x)] = (0.974836209 -0.222922333) ............... [sin(x)^2 + cos(x)^2] = 1
|>   2 [sin(x), cos(x)] = (-0.69576933 0.718265299) ............... [sin(x)^2 + cos(x)^2] = 1
|>   3 [sin(x), cos(x)] = (0.853703285 -0.520759734) ............... [sin(x)^2 + cos(x)^2] = 1
|>   4 [sin(x), cos(x)] = (-0.941210424 0.337820866) ............... [sin(x)^2 + cos(x)^2] = 1
|>   5 [sin(x), cos(x)] = (-0.8711111 -0.491085992) ............... [sin(x)^2 + cos(x)^2] = 1
|>   6 [sin(x), cos(x)] = (-0.0704802093 -0.997513178) ............... [sin(x)^2 + cos(x)^2] = 1
|>   7 [sin(x), cos(x)] = (0.490961891 -0.87118105) ............... [sin(x)^2 + cos(x)^2] = 1
|>   8 [sin(x), cos(x)] = (-0.78756139 0.616236202) ............... [sin(x)^2 + cos(x)^2] = 1
|>   9 [sin(x), cos(x)] = (0.715995247 0.698105154) ............... [sin(x)^2 + cos(x)^2] = 1

```
 */


GInt_t main(int argv, char** argc)
{

	// number of entries or trials
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
	//auto R2_lambda = [] __host__ __device__( thrust::tuple<GReal_t, GReal_t>* x )
	//{ return thrust::get<0>(*x)*thrust::get<0>(*x) +  thrust::get<1>(*x)*thrust::get<1>(*x); };

	auto R2_lambda = [] __host__ __device__( GReal_t* x )
	{ return x[0]*x[0] + x[1]*x[1]; };




	//Wrap the lambda
	auto R2_lambdaW  = wrap_lambda(R2_lambda);

	//Aggregate the functors in a tuple
   	auto functors = thrust::make_tuple( sin_lambaW, cos_lambaW);

    //start time
	auto start1 = std::chrono::high_resolution_clock::now();
	//Evaluate sin and cos
	auto result = eval(hydra::omp::sys , functors, angles_d.begin(), angles_d.end());
	//end time
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
	//time
	std::cout << "--------------------------------------------------------------"<<std::endl;
	std::cout << "| Evaluation of [sin(x), cos(x)] "<<std::endl;
	std::cout << "| Time (ms) = "<< elapsed1.count() <<std::endl;
	std::cout << "--------------------------------------------------------------"<<std::endl;


    //start time
    auto start2 = std::chrono::high_resolution_clock::now();
    //Evaluate {cos(angle)}^2 + {sin(angle)}^2
    auto result2 = eval(hydra::omp::sys , R2_lambdaW,result.begin(), result.end() );
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
