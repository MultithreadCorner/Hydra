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

#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/random.h>

#include <hydra/detail/Config.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/host_vector.h>
#include <hydra/multivector.h>



using namespace std;
using namespace hydra;





struct AccessOneA
{
	template<typename T>
	__host__ __device__ void operator()(T& e)
	{
		thrust::default_random_engine rng(thrust::default_random_engine::default_seed);
		thrust::uniform_real_distribution<float> UniRng(0.0f, 1.0f);

		e = UniRng(rng);

	}

};

template<unsigned int I >
struct AccessOneB
{
	template<typename T>
	__host__ __device__ void operator()(T& e)
	{
		thrust::default_random_engine rng(thrust::default_random_engine::default_seed);
		thrust::uniform_real_distribution<float> UniRng(0.0f, 1.0f);

		float x = UniRng(rng);
		thrust::get<I>(e) = x;

	}

};


struct AccessAllA
{

	template<typename T>
	__host__ __device__ void operator()(T t)
	{
		thrust::default_random_engine rng(thrust::default_random_engine::default_seed);
		thrust::uniform_real_distribution<float> UniRng(0.0f, 1.0f);


		float x= thrust::get<3>(t) + 2*UniRng(rng);
		float y= thrust::get<4>(t) + 2*UniRng(rng);
		float z= thrust::get<5>(t) + 2*UniRng(rng);

		float r     = sqrt( x*x + y*y + z*z);
		float theta = acos(z/r);
		float phi   = atan(y/x);

		thrust::get<0>(t) = r;
		thrust::get<1>(t) = theta;
		thrust::get<2>(t) = phi;

		thrust::get<3>(t) = x;
		thrust::get<4>(t) = y;
		thrust::get<5>(t) = z;

	}

};

struct AccessAllB
{

	template<typename T>
	__host__ __device__ void operator()(T& t)
	{
		thrust::default_random_engine rng(thrust::default_random_engine::default_seed);
		thrust::uniform_real_distribution<float> UniRng(0.0f, 1.0f);


		float x= thrust::get<3>(t) + 2*UniRng(rng);
		float y= thrust::get<4>(t) + 2*UniRng(rng);
		float z= thrust::get<5>(t) + 2*UniRng(rng);

		float r     = sqrt( x*x + y*y + z*z);
		float theta = acos(z/r);
		float phi   = atan(y/x);

		thrust::get<0>(t) = r;
		thrust::get<1>(t) = theta;
		thrust::get<2>(t) = phi;

		thrust::get<3>(t) = x;
		thrust::get<4>(t) = y;
		thrust::get<5>(t) = z;

	}

};


template<int I=0, typename T>
inline float _for_each_AccessOne1(T& storage)
{
	auto begin = storage.template vbegin<I>();
	auto end   = storage.template vend<I>();

	auto start1 = std::chrono::high_resolution_clock::now();
	thrust::for_each( begin, end, AccessOneA() );
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> elapsed1 = end1 - start1;
	return elapsed1.count();
}

template<int I=0,typename T>
inline float _for_each_AccessOne2(T& storage)
{
	auto start1 = std::chrono::high_resolution_clock::now();
	thrust::for_each(storage.begin(), storage.end(), AccessOneB<I>() );
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> elapsed1 = end1 - start1;
	return elapsed1.count();
}


template<typename T>
inline float _for_each1(T& storage)
{
	auto start1 = std::chrono::high_resolution_clock::now();
	thrust::for_each(storage.begin(), storage.end(), AccessAllA() );
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> elapsed1 = end1 - start1;
	return elapsed1.count();
}

template<typename T>
inline float _for_each2(T& storage)
{
	auto start1 = std::chrono::high_resolution_clock::now();
	thrust::for_each(storage.begin(), storage.end(), AccessAllB() );
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> elapsed1 = end1 - start1;
	return elapsed1.count();
}



size_t n=10000000;

int main(int argv, char** argc)
{

	typedef multivector<thrust::device_vector,
			thrust::device_malloc_allocator,float, float, float,float, float, float> table_t;

	typedef thrust::device_vector<thrust::tuple<float, float, float, float, float, float>> vector_t;
    thrust::tuple<float, float, float, float, float, float> init(0.0,0.0,0.0,2.0,2.0,2.0);

	{


		table_t  storage(n, init);
		float t=_for_each1(storage);
		std::cout << "--------------------------------------------------------------"<<std::endl;
		std::cout << "| multivector (acces all) "<<std::endl;
		std::cout << "| Time (ms) = "<< t <<std::endl;//elapsed1.count() <<std::endl;
		std::cout << "--------------------------------------------------------------"<<std::endl;
		 t=_for_each_AccessOne1<1, table_t >(storage);
		std::cout << "--------------------------------------------------------------"<<std::endl;
		std::cout << "| multivector (access one)"<<std::endl;
		std::cout << "| Time (ms) = "<< t <<std::endl;//elapsed1.count() <<std::endl;
		std::cout << "--------------------------------------------------------------"<<std::endl;

		for(size_t i=0; i<10; i++)
			std::cout<< storage[i] << std::endl;



	}

	//---
	{
		vector_t  storage(n, init);
		//start time
		//auto start1 = std::chrono::high_resolution_clock::now();
		float t= _for_each2(storage);
		//auto end1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed1 = end1 - start1;
		//time
		std::cout << "--------------------------------------------------------------"<<std::endl;
		std::cout << "| vector (acces all) "<<std::endl;
		std::cout << "| Time (ms) = "<<  t <<std::endl;//elapsed1.count() <<std::endl;
		std::cout << "--------------------------------------------------------------"<<std::endl;
		t=_for_each_AccessOne2<1,vector_t >(storage);
		std::cout << "--------------------------------------------------------------"<<std::endl;
		std::cout << "| vector (access one)"<<std::endl;
		std::cout << "| Time (ms) = "<< t <<std::endl;//elapsed1.count() <<std::endl;
		std::cout << "--------------------------------------------------------------"<<std::endl;
		for(size_t i=0; i<10; i++)
					std::cout<< storage[i] << std::endl;


	}



}


