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
 * multivector_testing.cpp
 *
 *  Created on: 21/10/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef MULTIVECTOR_TESTING_CPP_
#define MULTIVECTOR_TESTING_CPP_

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch/catch.hpp"

#include <memory>
#include <limits>

#include <hydra/experimental/multivector.h>

using namespace std;
using namespace hydra;


TEST_CASE( "multivector<thrust::host_vector, std::allocator, unsigned int, float, double> ",
		"[hydra::multivector]" ) {

	typedef experimental::multivector<thrust::device_vector, thrust::device_malloc_allocator, unsigned int, float, double> table_t;

size_t N=1000;

	SECTION( "multivector(): size, capacity, emptiness, resize, reserve" )
	{
		table_t table;
		REQUIRE( table.empty() == true );
		REQUIRE( table.size()  == 0 );
		REQUIRE( table.capacity() == 0 );

		table.resize(10);

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10  );

		table.reserve(N);

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= N );

	}

	SECTION( "multivector(size_t n): size, capacity, emptiness, resize, reserve" )
	{
		table_t table(10);
		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

		table.resize(N);

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == N );
		REQUIRE( table.capacity() >= N );

		table.reserve(N);

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == N );
		REQUIRE( table.capacity() >= N );
	}

	SECTION( "multivector(multivector other ): size, capacity, emptiness, resize, reserve" )
	{
		table_t table(N);
		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == N );
		REQUIRE( table.capacity() >= N );

		for(int i =0; i< table.size(); i++ ){
			table[i] = thrust::make_tuple(i,
					i+std::numeric_limits<float>::lowest(),
					i+std::numeric_limits<double>::lowest());
		}

		table_t other( table);

		REQUIRE( other.empty() == false );
		REQUIRE( other.size()  == N );
		REQUIRE( other.capacity() >= N );

		for(int i =0; i< table.size(); i++ ){
			REQUIRE( thrust::get<0>(table[i]) ==  Approx(thrust::get<0>(other[i])) );
			REQUIRE( thrust::get<1>(table[i]) ==  Approx(thrust::get<1>(other[i])) );
			REQUIRE( thrust::get<2>(table[i]) ==  Approx(thrust::get<2>(other[i])) );
		}
	}

	SECTION( "multivector: push_back, pop_back,front, back" )
		{
			table_t table1;
			REQUIRE( table1.empty() == true );
			REQUIRE( table1.size()  == 0 );
			REQUIRE( table1.capacity() >= 0 );

			table_t table2(N);

			for(int i =0; i< N; i++ ){

				table1.push_back(thrust::make_tuple( i,
						i+std::numeric_limits<float>::lowest(),
						i+std::numeric_limits<double>::lowest()));

				table2[i] = thrust::make_tuple( i,
							i+std::numeric_limits<float>::lowest(),
							i+std::numeric_limits<double>::lowest());
			}

			REQUIRE( table1.empty() == false );
			REQUIRE( table1.size()  == N );
			REQUIRE( table1.capacity() >= N );

			for(int i =0; i< N; i++ ){
				REQUIRE( thrust::get<0>(table1[i]) ==  Approx(thrust::get<0>(table2[i])) );
				REQUIRE( thrust::get<1>(table1[i]) ==  Approx(thrust::get<1>(table2[i])) );
				REQUIRE( thrust::get<2>(table1[i]) ==  Approx(thrust::get<2>(table2[i])) );
			}

			for(int i =0; i< N; i++ )
				table1.pop_back();

			REQUIRE( table1.empty() == true );
			REQUIRE( table1.size()  == 0 );
			REQUIRE( table1.capacity() >= N );

			for(int i =0; i< N; i++ ){

				table1.push_back(i,	i+std::numeric_limits<float>::lowest(),
						i+std::numeric_limits<double>::lowest());
			}

			REQUIRE( table1.empty() == false );
			REQUIRE( table1.size()  == N );
			REQUIRE( table1.capacity() >= N );

			for(int i =0; i< N; i++ ){
				REQUIRE( thrust::get<0>(table1[i]) ==  Approx(thrust::get<0>(table2[i])) );
				REQUIRE( thrust::get<1>(table1[i]) ==  Approx(thrust::get<1>(table2[i])) );
				REQUIRE( thrust::get<2>(table1[i]) ==  Approx(thrust::get<2>(table2[i])) );
			}

			REQUIRE( table1.front() == table2.front() );

			table1.front()=thrust::make_tuple( 0,1,2);

			REQUIRE( thrust::get<0>(table1[0]) ==  0 );
			REQUIRE( thrust::get<1>(table1[0]) ==  1 );
			REQUIRE( thrust::get<2>(table1[0]) ==  2 );

			REQUIRE( table1.back() == table2.back() );

			table1.back()=thrust::make_tuple( 0,1,2);

			REQUIRE( thrust::get<0>(table1[N-1]) ==  0 );
			REQUIRE( thrust::get<1>(table1[N-1]) ==  1 );
			REQUIRE( thrust::get<2>(table1[N-1]) ==  2 );

		}

	SECTION( "multivector: iterators" )
		{
			table_t table;
			REQUIRE( table.empty() == true );
			REQUIRE( table.size()  == 0 );
			REQUIRE( table.capacity() >= 0 );

			//fill
			for(int i =0; i< N; i++ ){

				table.push_back(i,i,i);
			}


			//test c++11 semantics

			int i=0;
			for(auto row:table )
			{
				REQUIRE( thrust::get<0>(row) == i );
				REQUIRE( thrust::get<1>(row) == Approx(i) );
				REQUIRE( thrust::get<2>(row) == Approx(i) );

				i++;
			}

			//direct iterators
			i=0;
			for(auto row = table.begin(); row!= table.end(); row++)
			{
				REQUIRE( thrust::get<0>(*row) == i );
				REQUIRE( thrust::get<1>(*row) == Approx(i) );
				REQUIRE( thrust::get<2>(*row) == Approx(i) );

				i++;
			}

			i=0;
			for(auto row = table.cbegin(); row!= table.cend(); row++)
			{
				REQUIRE( thrust::get<0>(*row) == i );
				REQUIRE( thrust::get<1>(*row) == Approx(i) );
				REQUIRE( thrust::get<2>(*row) == Approx(i) );

				i++;
			}

			//reverse iterators
			i=N-1;
			for(auto row = table.rbegin(); row!= table.rend(); row++)
			{
				REQUIRE( thrust::get<0>(*row) == i );
				REQUIRE( thrust::get<1>(*row) == Approx(i) );
				REQUIRE( thrust::get<2>(*row) == Approx(i) );

				i--;
			}

			i=N-1;
			for(auto row = table.crbegin(); row!= table.crend(); row++)
			{
				REQUIRE( thrust::get<0>(*row) == i );
				REQUIRE( thrust::get<1>(*row) == Approx(i) );
				REQUIRE( thrust::get<2>(*row) == Approx(i) );

				i--;
			}


		}

	SECTION( "multivector: clear, capacity, shrink_to_fit" )
		{
			table_t table;
			REQUIRE( table.empty() == true );
			REQUIRE( table.size()  == 0 );
			REQUIRE( table.capacity() >= 0 );

			//fill
			for(int i =0; i< N; i++ ){

				table.push_back(i,i,i);
			}

			table.clear();

			REQUIRE( table.empty() == true );
			REQUIRE( table.size()  == 0 );
			REQUIRE( table.capacity() >= N );

			table.shrink_to_fit();

			REQUIRE( table.empty() == true );
			REQUIRE( table.size()  == 0 );
			REQUIRE_FALSE( table.capacity() >= N );

		}

}


#endif /* MULTIVECTOR_TESTING_CPP_ */
