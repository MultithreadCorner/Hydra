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
 * multivector_performance.cu
 *
 *  Created on: 23/10/2016
 *      Author: Antonio Augusto Alves Junior
 */



#define NONIUS_RUNNER
#include <nonius/nonius.h++>

#include <memory>
#include <limits>

#include <hydra/multivector.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>


typedef multivector<thrust::host_vector,
		std::allocator,	unsigned int, float, double> table_t;

typedef thrust::host_vector<thrust::tuple<unsigned int, float, double>> vector_t;


struct change
{
	__host__ __device__
	template<typename T>
	void operator()(T t){ thrust::get<1>(t)=1.0; }

};

template<typename T>
void _for_each(T& storage)
{

	thrust::for_each(storage.begin(), storage.end(), change() );


}

template<size_t E, typename T>
double reduce(T& storage)
{
	double sum=0.0;

	for( size_t i=0; i<storage.size()-1;i++){
		sum += thrust::get<E>( storage[i])
			+  thrust::get<E>( storage[i+1]);
		}

	return sum;
}



NONIUS_PARAM(size, 100000u)
NONIUS_PARAM(tuple, thrust::tuple<unsigned int, float, double>(0,0,0 ))

/**
 * multivector<thrust::host_vector, std::allocator,unsigned int, float, double>
 */
NONIUS_BENCHMARK("multivector<thrust::host_vector, std::allocator,unsigned int, float, double>::push_back",
		[](nonius::parameters params) {
	auto n = params.get<size>();
	auto t = params.get<tuple>();
	return [=](nonius::chronometer meter) {
		table_t  storage;
		meter.measure([&]() {
			for (auto j = 0u; j < n; ++j)
				storage.push_back(t);
		});
	};
})

/**
 * multivector<thrust::host_vector, std::allocator,unsigned int, float, double>
 */
NONIUS_BENCHMARK("multivector<thrust::host_vector, std::allocator,unsigned int, float, double>:: access+set",
		[](nonius::parameters params) {
	auto n = params.get<size>();
	auto t = params.get<tuple>();
	return [=](nonius::chronometer meter) {
		table_t  storage;
		for (auto j = 0u; j < n; ++j)
			storage.push_back(t);

		meter.measure([&]() { _for_each(storage);});

	};
})


/**
 * thrust::host_vector<thrust::tuple<unsigned int, float, double>>::push_back
 */
NONIUS_BENCHMARK("thrust::host_vector<thrust::tuple<unsigned int, float, double>>::push_back",
		[](nonius::parameters params) {
	auto n = params.get<size>();
	auto t = params.get<tuple>();
	return [=](nonius::chronometer meter) {
		vector_t  storage;
		meter.measure([&]() {

			for (auto j = 0u; j < n; ++j)
				storage.push_back(t);
		});
	};
})



/**
 * thrust::host_vector<thrust::tuple<unsigned int, float, double>>::push_back
 */
NONIUS_BENCHMARK("thrust::host_vector<thrust::tuple<unsigned int, float, double>>:: access+set",
		[](nonius::parameters params) {
	auto n = params.get<size>();
	auto t = params.get<tuple>();
	return [=](nonius::chronometer meter) {
		vector_t  storage;

		for (auto j = 0u; j < n; ++j)
				storage.push_back(t);

		meter.measure([&]() { _for_each(storage);});

	};
})














