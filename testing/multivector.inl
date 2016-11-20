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

#pragma once

#include <memory>
#include <limits>
#include <utility>

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/experimental/multivector.h>


TEST_CASE( "multivector","hydra::multivector" ) {

	typedef thrust::tuple<unsigned int, float, double> tuple_t;

	typedef thrust::host_vector<tuple_t>   mvector_h;
	typedef thrust::device_vector<tuple_t> mvector_d;

	typedef hydra::experimental::multivector<mvector_h> table_h;
	typedef hydra::experimental::multivector<mvector_d> table_d;

	SECTION( "default constructor <host backend> : size, capacity, emptiness, resize, reserve" )
	{
		table_h table;
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

	SECTION( "default constructor <device backend> : size, capacity, emptiness, resize, reserve" )
		{
			table_d table;
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

	SECTION( "multivector(size_t n) <host backend> : size, capacity, emptiness, resize, reserve" )
	{
		table_h table(10);
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

	SECTION( "multivector(size_t n) <device backend>: size, capacity, emptiness, resize, reserve" )
	{
		table_d table(10);
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



	SECTION( "copy constructor <host backend> multivector(other): size, capacity, emptiness, resize, reserve" )
	{
		table_h table(10);
		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

		for(int i =0; i< table.size(); i++ ){
			table[i] = thrust::make_tuple(i,i,i);
		}

		table_h other( table);

		REQUIRE( other.empty() == false );
		REQUIRE( other.size()  == 10 );
		REQUIRE( other.capacity() >= 10 );

		for(int i =0; i< table.size(); i++ ){
			REQUIRE( thrust::get<0>(table[i]) ==  Approx(thrust::get<0>(other[i])) );
			REQUIRE( thrust::get<1>(table[i]) ==  Approx(thrust::get<1>(other[i])) );
			REQUIRE( thrust::get<2>(table[i]) ==  Approx(thrust::get<2>(other[i])) );
		}
	}

	SECTION( "copy constructor <device backend> multivector(other): size, capacity, emptiness, resize, reserve" )
	{
		table_d table(10);
		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

		for(int i =0; i< table.size(); i++ ){
			table[i] = thrust::make_tuple(i,i,i);
		}

		table_d other( table);

		REQUIRE( other.empty() == false );
		REQUIRE( other.size()  == 10 );
		REQUIRE( other.capacity() >= 10 );

		for(int i =0; i< table.size(); i++ ){
			REQUIRE( thrust::get<0>(table[i]) ==  Approx(thrust::get<0>(other[i])) );
			REQUIRE( thrust::get<1>(table[i]) ==  Approx(thrust::get<1>(other[i])) );
			REQUIRE( thrust::get<2>(table[i]) ==  Approx(thrust::get<2>(other[i])) );
		}
	}

	SECTION( "copy constructor <device -> host backend> multivector(other): size, capacity, emptiness, resize, reserve" )
		{
			table_d table(10);
			REQUIRE( table.empty() == false );
			REQUIRE( table.size()  == 10 );
			REQUIRE( table.capacity() >= 10 );

			for(int i =0; i< table.size(); i++ ){
				table[i] = thrust::make_tuple(i,i,i);
			}

			table_h other( table);

			REQUIRE( other.empty() == false );
			REQUIRE( other.size()  == 10 );
			REQUIRE( other.capacity() >= 10 );

			for(int i =0; i< table.size(); i++ ){
				REQUIRE( thrust::get<0>(table[i]) ==  Approx(thrust::get<0>(other[i])) );
				REQUIRE( thrust::get<1>(table[i]) ==  Approx(thrust::get<1>(other[i])) );
				REQUIRE( thrust::get<2>(table[i]) ==  Approx(thrust::get<2>(other[i])) );
			}
		}

	SECTION( "copy constructor <host -> device backend> multivector(other): size, capacity, emptiness, resize, reserve" )
	{
		table_h table(10);
		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

		for(int i =0; i< table.size(); i++ ){
			table[i] = thrust::make_tuple(i,i,i);
		}

		table_d other( table);

		REQUIRE( other.empty() == false );
		REQUIRE( other.size()  == 10 );
		REQUIRE( other.capacity() >= 10 );

		for(int i =0; i< table.size(); i++ ){
			REQUIRE( thrust::get<0>(table[i]) ==  Approx(thrust::get<0>(other[i])) );
			REQUIRE( thrust::get<1>(table[i]) ==  Approx(thrust::get<1>(other[i])) );
			REQUIRE( thrust::get<2>(table[i]) ==  Approx(thrust::get<2>(other[i])) );
		}
	}


	SECTION( "move constructor multivector(other) <host backend> : size, capacity, emptiness, resize, reserve" )
	{
		table_h table(10);
		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

		for(int i =0; i< table.size(); i++ ){
			table[i] = thrust::make_tuple(i,i,i);
		}

		table_h other( std::move(table));

		REQUIRE( other.empty() == false );
		REQUIRE( other.size()  == 10 );
		REQUIRE( other.capacity() >= 10 );

		REQUIRE( table.empty() == true );
		REQUIRE( table.size()  == 0 );
		REQUIRE( table.capacity() == 0 );

		for(int i =0; i< 10; i++ ){

			table.push_back(thrust::make_tuple( i+1, (i+1.0),(i+1.0)));

		}

		for(int i =0; i< 10; i++ ){

			REQUIRE( thrust::get<0>(table[i]) !=  Approx(thrust::get<0>(other[i])) );
			REQUIRE( thrust::get<1>(table[i]) !=  Approx(thrust::get<1>(other[i])) );
			REQUIRE( thrust::get<2>(table[i]) !=  Approx(thrust::get<2>(other[i])) );
		}

		for(int i =0; i< 10; i++ ){

			REQUIRE( thrust::get<0>(table[i]) ==  Approx(i+1)) ;
			REQUIRE( thrust::get<1>(table[i]) ==  Approx(i+1)) ;
			REQUIRE( thrust::get<2>(table[i]) ==  Approx(i+1)) ;
		}

		for(int i =0; i< 10; i++ ){

			REQUIRE( thrust::get<0>(other[i]) ==  Approx(i)) ;
			REQUIRE( thrust::get<1>(other[i]) ==  Approx(i)) ;
			REQUIRE( thrust::get<2>(other[i]) ==  Approx(i)) ;
		}

	}

	SECTION( "move constructor multivector(other) <device backend> : size, capacity, emptiness, resize, reserve" )
		{
			table_d table(10);
			REQUIRE( table.empty() == false );
			REQUIRE( table.size()  == 10 );
			REQUIRE( table.capacity() >= 10 );

			for(int i =0; i< table.size(); i++ ){
				table[i] = thrust::make_tuple(i,i,i);
			}

			table_d other( std::move(table));

			REQUIRE( other.empty() == false );
			REQUIRE( other.size()  == 10 );
			REQUIRE( other.capacity() >= 10 );

			REQUIRE( table.empty() == true );
			REQUIRE( table.size()  == 0 );
			REQUIRE( table.capacity() == 0 );

			for(int i =0; i< 10; i++ ){

				table.push_back(thrust::make_tuple( i+1, (i+1.0),(i+1.0)));

			}

			for(int i =0; i< 10; i++ ){

				REQUIRE( thrust::get<0>(table[i]) !=  Approx(thrust::get<0>(other[i])) );
				REQUIRE( thrust::get<1>(table[i]) !=  Approx(thrust::get<1>(other[i])) );
				REQUIRE( thrust::get<2>(table[i]) !=  Approx(thrust::get<2>(other[i])) );
			}

			for(int i =0; i< 10; i++ ){

				REQUIRE( thrust::get<0>(table[i]) ==  Approx(i+1)) ;
				REQUIRE( thrust::get<1>(table[i]) ==  Approx(i+1)) ;
				REQUIRE( thrust::get<2>(table[i]) ==  Approx(i+1)) ;
			}

			for(int i =0; i< 10; i++ ){

				REQUIRE( thrust::get<0>(other[i]) ==  Approx(i)) ;
				REQUIRE( thrust::get<1>(other[i]) ==  Approx(i)) ;
				REQUIRE( thrust::get<2>(other[i]) ==  Approx(i)) ;
			}

		}

	SECTION( "<host backend> push_back, pop_back, front and back" )
	{
		table_h table1;
		REQUIRE( table1.empty() == true );
		REQUIRE( table1.size()  == 0 );
		REQUIRE( table1.capacity() >= 0 );

		table_h table2(10);

		for(int i =0; i< 10; i++ ){

			table1.push_back(thrust::make_tuple( i,i,i));

			table2[i] = thrust::make_tuple( i,i,i);
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

			table1.push_back(i,	i,i);
		}

		REQUIRE( table1.empty() == false );
		REQUIRE( table1.size()  == 10 );
		REQUIRE( table1.capacity() >= 10 );

		for(int i =0; i< 10; i++ ){
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

		REQUIRE( thrust::get<0>(table1[9]) ==  0 );
		REQUIRE( thrust::get<1>(table1[9]) ==  1 );
		REQUIRE( thrust::get<2>(table1[9]) ==  2 );

	}


	SECTION( "<device backend> push_back, pop_back, front and back" )
	{
		table_d table1;

		REQUIRE( table1.empty() == true );
		REQUIRE( table1.size()  == 0 );
		REQUIRE( table1.capacity() >= 0 );

		table_d table2(10);

		for(int i =0; i< 10; i++ ){

			table1.push_back(thrust::make_tuple( i,i,i));

			table2[i] = thrust::make_tuple( i,i,i);
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

			table1.push_back(i,	i,i);
		}

		REQUIRE( table1.empty() == false );
		REQUIRE( table1.size()  == 10 );
		REQUIRE( table1.capacity() >= 10 );

		for(int i =0; i< 10; i++ ){
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

		REQUIRE( thrust::get<0>(table1[9]) ==  0 );
		REQUIRE( thrust::get<1>(table1[9]) ==  1 );
		REQUIRE( thrust::get<2>(table1[9]) ==  2 );

	}

	SECTION( "iterators <host backend>" )
	{
		table_h table;
		REQUIRE( table.empty() == true );
		REQUIRE( table.size()  == 0 );
		REQUIRE( table.capacity() >= 0 );

		//fill
		for(int i =0; i< 10; i++ ){

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
		i=9;
		for(auto row = table.rbegin(); row!= table.rend(); row++)
		{
			REQUIRE( thrust::get<0>(*row) == i );
			REQUIRE( thrust::get<1>(*row) == Approx(i) );
			REQUIRE( thrust::get<2>(*row) == Approx(i) );

			i--;
		}

		i=9;
		for(auto row = table.crbegin(); row!= table.crend(); row++)
		{
			REQUIRE( thrust::get<0>(*row) == i );
			REQUIRE( thrust::get<1>(*row) == Approx(i) );
			REQUIRE( thrust::get<2>(*row) == Approx(i) );

			i--;
		}


	}

	SECTION( "iterators <device backend>" )
		{
			table_d table;
			REQUIRE( table.empty() == true );
			REQUIRE( table.size()  == 0 );
			REQUIRE( table.capacity() >= 0 );

			//fill
			for(int i =0; i< 10; i++ ){

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
			i=9;
			for(auto row = table.rbegin(); row!= table.rend(); row++)
			{
				REQUIRE( thrust::get<0>(*row) == i );
				REQUIRE( thrust::get<1>(*row) == Approx(i) );
				REQUIRE( thrust::get<2>(*row) == Approx(i) );

				i--;
			}

			i=9;
			for(auto row = table.crbegin(); row!= table.crend(); row++)
			{
				REQUIRE( thrust::get<0>(*row) == i );
				REQUIRE( thrust::get<1>(*row) == Approx(i) );
				REQUIRE( thrust::get<2>(*row) == Approx(i) );

				i--;
			}


		}



	SECTION( "<host backend> clear, capacity, shrink_to_fit" )
	{
		table_h table;
		REQUIRE( table.empty() == true );
		REQUIRE( table.size()  == 0 );
		REQUIRE( table.capacity() >= 0 );

		//fill
		for(int i =0; i< 10; i++ ){

			table.push_back(i,i,i);
		}

		table.clear();

		REQUIRE( table.empty() == true );
		REQUIRE( table.size()  == 0 );
		REQUIRE( table.capacity() >= 10 );

		table.shrink_to_fit();

		REQUIRE( table.empty() == true );
		REQUIRE( table.size()  == 0 );
		REQUIRE_FALSE( table.capacity() >= 10 );

	}

	SECTION( "<device backend> clear, capacity, shrink_to_fit" )
	{
		table_d table;
		REQUIRE( table.empty() == true );
		REQUIRE( table.size()  == 0 );
		REQUIRE( table.capacity() >= 0 );

		//fill
		for(int i =0; i< 10; i++ ){

			table.push_back(i,i,i);
		}

		table.clear();

		REQUIRE( table.empty() == true );
		REQUIRE( table.size()  == 0 );
		REQUIRE( table.capacity() >= 10 );

		table.shrink_to_fit();

		REQUIRE( table.empty() == true );
		REQUIRE( table.size()  == 0 );
		REQUIRE_FALSE( table.capacity() >= 10 );

	}
}

