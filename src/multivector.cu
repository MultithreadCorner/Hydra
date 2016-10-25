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

#include <chrono>
#include <time.h>
#include <hydra/detail/Config.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/host_vector.h>
#include <hydra/experimental/multivector.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>

using namespace std;
using namespace hydra;

struct change
{

	template<typename T>
	__host__ __device__ void operator()(T t){
		thrust::get<0>(t)= sqrt((double)sin((double)thrust::get<2>(t)));


	}

};


template<typename T>
void _for_each(T& storage)
{
	thrust::for_each(storage.begin(), storage.end(), change() );
}

size_t n=1000;

int main(int argv, char** argc)
{

	typedef hydra::experimental::multivector<thrust::device_vector,
			thrust::device_malloc_allocator,
			unsigned int, float, double> table_t;

	typedef thrust::device_vector<thrust::tuple<unsigned int, float, double>> vector_t;

	{
		table_t  storage;
		for (auto j = 0u; j < n; ++j)
			storage.push_back(j,j,j);

		//start time
		auto start1 = std::chrono::high_resolution_clock::now();
		_for_each(storage);
		auto end1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
		//time
		std::cout << "--------------------------------------------------------------"<<std::endl;
		std::cout << "| multivector "<<std::endl;
		std::cout << "| Time (ms) = "<< elapsed1.count() <<std::endl;
		std::cout << "--------------------------------------------------------------"<<std::endl;

	}
	{
		vector_t  storage;
		for (auto j = 0u; j < n; ++j)
			storage.push_back(thrust::make_tuple(j,j,j));

		//start time
		auto start1 = std::chrono::high_resolution_clock::now();
		_for_each(storage);
		auto end1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
		//time
		std::cout << "--------------------------------------------------------------"<<std::endl;
		std::cout << "| vector "<<std::endl;
		std::cout << "| Time (ms) = "<< elapsed1.count() <<std::endl;
		std::cout << "--------------------------------------------------------------"<<std::endl;

	}
}
