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

	typedef experimental::multivector<thrust::host_vector, std::allocator, unsigned int, float, double> table_t;



	SECTION( "multivector(): size, capacity, emptiness, resize, reserve" )
	{
		table_t table;
		REQUIRE( table.empty() == true );
		REQUIRE( table.size()  == 0 );
		REQUIRE( table.capacity() == 0 );

		table.resize(10);

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

		table.reserve(20);

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 20 );
	}

	SECTION( "multivector(size_t n): size, capacity, emptiness, resize, reserve" )
	{
		table_t table(10);
		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

		table.resize(20);

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 20 );
		REQUIRE( table.capacity() >= 20 );

		table.reserve(30);

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 20 );
		REQUIRE( table.capacity() >= 30 );
	}

	SECTION( "multivector(multivector other ): size, capacity, emptiness, resize, reserve" )
	{
		table_t table(10);
		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

		for(int i =0; i< table.size(); i++ ){
			table[i] = thrust::make_tuple(i,
					i+std::numeric_limits<float>::lowest(),
					i+std::numeric_limits<double>::lowest());
		}

		table_t other( table);

		REQUIRE( other.empty() == false );
		REQUIRE( other.size()  == 10 );
		REQUIRE( other.capacity() >= 10 );

		for(int i =0; i< table.size(); i++ ){
			REQUIRE( thrust::get<0>(table[i]) ==  Approx(thrust::get<0>(other[i])) );
			REQUIRE( thrust::get<1>(table[i]) ==  Approx(thrust::get<1>(other[i])) );
			REQUIRE( thrust::get<2>(table[i]) ==  Approx(thrust::get<2>(other[i])) );
		}
	}

	SECTION( "multivector(multivector other ): push_back, pop_back,front, back, data" )
		{
			table_t table1;
			REQUIRE( table1.empty() == true );
			REQUIRE( table1.size()  == 0 );
			REQUIRE( table1.capacity() >= 0 );

			table_t table2(10);

			for(int i =0; i< 10; i++ ){

				table1.push_back(thrust::make_tuple( i,
						i+std::numeric_limits<float>::lowest(),
						i+std::numeric_limits<double>::lowest()));

				table2[i] = thrust::make_tuple( i,
							i+std::numeric_limits<float>::lowest(),
							i+std::numeric_limits<double>::lowest());
			}

			REQUIRE( table1.empty() == false );
			REQUIRE( table1.size()  == 10 );
			REQUIRE( table1.capacity() >= 10 );

			for(int i =0; i< 10; i++ ){
				REQUIRE( thrust::get<0>(table1[i]) ==  Approx(thrust::get<0>(table2[i])) );
				REQUIRE( thrust::get<1>(table1[i]) ==  Approx(thrust::get<1>(table2[i])) );
				REQUIRE( thrust::get<2>(table1[i]) ==  Approx(thrust::get<2>(table2[i])) );
			}

			for(int i =0; i< 10; i++ )
				table1.pop_back();

			REQUIRE( table1.empty() == true );
			REQUIRE( table1.size()  == 0 );
			REQUIRE( table1.capacity() >= 10 );

			for(int i =0; i< 10; i++ ){

				table1.push_back(i,	i+std::numeric_limits<float>::lowest(),
						i+std::numeric_limits<double>::lowest());
			}

			REQUIRE( table1.empty() == false );
			REQUIRE( table1.size()  == 10 );
			REQUIRE( table1.capacity() >= 10 );

			for(int i =0; i< 10; i++ ){
				REQUIRE( thrust::get<0>(table1[i]) ==  Approx(thrust::get<0>(table2[i])) );
				REQUIRE( thrust::get<1>(table1[i]) ==  Approx(thrust::get<1>(table2[i])) );
				REQUIRE( thrust::get<2>(table1[i]) ==  Approx(thrust::get<2>(table2[i])) );
			}



		}

}


#endif /* MULTIVECTOR_TESTING_CPP_ */
