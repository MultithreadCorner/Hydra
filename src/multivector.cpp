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
 * multivector.cpp
 *
 *  Created on: 19/10/2016
 *      Author: Antonio Augusto Alves Junior
 */
#include <memory>
#include <chrono>
#include <time.h>

#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/random.h>

#include <hydra/detail/Config.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/host_vector.h>
#include <hydra/experimental/multivector.h>


using namespace std;
using namespace hydra;

struct ConvertA
{

	template<typename T>
	__host__ __device__ void operator()(T t)
	{
		thrust::default_random_engine rng(thrust::default_random_engine::default_seed);
		thrust::uniform_real_distribution<double> UniRng(0.0f, 1.0f);


		double x= UniRng(rng);
		double y= UniRng(rng);
		double z= UniRng(rng);

		double r     = sqrt( x*x + y*y + z*z);
		double theta = acos(z/r);
		double phi   = atan(y/x);

		thrust::get<0>(t) = r;
		thrust::get<1>(t) = theta;
		thrust::get<2>(t) = phi;

		thrust::get<3>(t) = x;
		thrust::get<4>(t) = y;
		thrust::get<5>(t) = z;

	}

};

struct ConvertB
{

	template<typename T>
	__host__ __device__ void operator()(T& t)
	{
		thrust::default_random_engine rng(thrust::default_random_engine::default_seed);
		thrust::uniform_real_distribution<double> UniRng(0.0f, 1.0f);


		double x= UniRng(rng);
		double y= UniRng(rng);
		double z= UniRng(rng);

		double r     = sqrt( x*x + y*y + z*z);
		double theta = acos(z/r);
		double phi   = atan(y/x);

		thrust::get<0>(t) = r;
		thrust::get<1>(t) = theta;
		thrust::get<2>(t) = phi;

		thrust::get<3>(t) = x;
		thrust::get<4>(t) = y;
		thrust::get<5>(t) = z;

	}

};




template<typename T>
double _for_each1(T& storage)
{
	auto start1 = std::chrono::high_resolution_clock::now();
	thrust::for_each(storage.begin(), storage.end(), ConvertA() );
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
	return elapsed1.count();
}

template<typename T>
double _for_each2(T& storage)
{
	auto start1 = std::chrono::high_resolution_clock::now();
	thrust::for_each(storage.begin(), storage.end(), ConvertB() );
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
	return elapsed1.count();
}



size_t n=100000000;

int main(int argv, char** argc)
{

	typedef hydra::experimental::multivector<thrust::host_vector,
			std::allocator,double, double, double,double, double, double> table_t;

	typedef thrust::host_vector<thrust::tuple<double, double, double,double, double, double>> vector_t;


	{
		table_t  storage(n);
		//start time
		//auto start1 = std::chrono::high_resolution_clock::now();
	    double t=_for_each1(storage);
		//auto end1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
		//time
		std::cout << "--------------------------------------------------------------"<<std::endl;
		std::cout << "| multivector "<<std::endl;
		std::cout << "| Time (ms) = "<< t <<std::endl;//elapsed1.count() <<std::endl;
		std::cout << "--------------------------------------------------------------"<<std::endl;
		for(size_t i=0; i<10; i++)
			std::cout<< storage[i] << std::endl;

	}

	//---
	{
		vector_t  storage(n);
		//start time
		//auto start1 = std::chrono::high_resolution_clock::now();
		double t= _for_each2(storage);
		//auto end1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
		//time
		std::cout << "--------------------------------------------------------------"<<std::endl;
		std::cout << "| vector "<<std::endl;
		std::cout << "| Time (ms) = "<<  t <<std::endl;//elapsed1.count() <<std::endl;
		std::cout << "--------------------------------------------------------------"<<std::endl;
		for(size_t i=0; i<10; i++)
					std::cout<< storage[i] << std::endl;

	}



}
