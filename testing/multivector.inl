/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
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

#include <hydra/multivector.h>
#include <hydra/Tuple.h>
#include <hydra/device/System.h>
#include <hydra/host/System.h>

using Catch::Approx;


TEST_CASE( "multivector","hydra::multivector" ) {

	typedef hydra::tuple<unsigned int, int, float, double> tuple_t;

	typedef hydra::multivector<	tuple_t, hydra::host::sys_t>   table_h;
	typedef hydra::multivector<	tuple_t, hydra::device::sys_t> table_d;

	SECTION( "default constructor" )
	{
		table_d table;
		REQUIRE( table.empty() == true );
		REQUIRE( table.size()  == 0 );
		REQUIRE( table.capacity() == 0 );

	}

	SECTION( "constructor multivector(size_t n)" )
	{
		table_d table(10);
		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

	}

	SECTION( "constructor multivector(size_t n, value_type x)" )
	{
		table_d table(10, tuple_t{1, -1, 1.0, 1.0});

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

		for(size_t i =0; i< table.size(); i++ ){
			REQUIRE( hydra::get<0>(table[i]) ==  1);
			REQUIRE( hydra::get<1>(table[i]) == -1);
			REQUIRE( hydra::get<2>(table[i]) ==  Approx(1.0) );
			REQUIRE( hydra::get<3>(table[i]) ==  Approx(1.0) );
		}
	}

	SECTION( "constructor multivector(hydra::pair<size_t,value_type>(n,x))" )
	{
		table_d table(hydra::pair<size_t,tuple_t>(10, tuple_t{1, -1, 1.0, 1.0}));

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

		for(size_t i =0; i< table.size(); i++ ){
			REQUIRE( hydra::get<0>(table[i]) ==  1);
			REQUIRE( hydra::get<1>(table[i]) == -1);
			REQUIRE( hydra::get<2>(table[i]) ==  Approx(1.0) );
			REQUIRE( hydra::get<3>(table[i]) ==  Approx(1.0) );
		}

	}


	SECTION( "copy constructor (same backend) multivector(other)" )
	{
		table_d table(10);

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

		for(size_t i =0; i< table.size(); i++ ){
			table[i] = hydra::make_tuple(i,i,i,i);
		}

		table_d other( table);

		REQUIRE( other.empty() == false );
		REQUIRE( other.size()  == 10 );
		REQUIRE( other.capacity() >= 10 );

		for(size_t i =0; i< table.size(); i++ ){
			REQUIRE( hydra::get<0>(table[i]) ==  (hydra::get<0>(other[i])) );
			REQUIRE( hydra::get<1>(table[i]) ==  (hydra::get<1>(other[i])) );
			REQUIRE( hydra::get<2>(table[i]) ==  Approx(hydra::get<2>(other[i])) );
			REQUIRE( hydra::get<2>(table[i]) ==  Approx(hydra::get<2>(other[i])) );
		}
	}

	SECTION( "copy constructor (iterators, same backend) multivector(other)" )
	{
		table_d table(10);

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

		for(size_t i =0; i< table.size(); i++ ){
			table[i] = hydra::make_tuple(i,i,i,i);
		}

		table_d other( table.begin(), table.end());

		REQUIRE( other.empty() == false );
		REQUIRE( other.size()  == 10 );
		REQUIRE( other.capacity() >= 10 );

		for(size_t i =0; i< table.size(); i++ ){
			REQUIRE( hydra::get<0>(table[i]) ==  (hydra::get<0>(other[i])) );
			REQUIRE( hydra::get<1>(table[i]) ==  (hydra::get<1>(other[i])) );
			REQUIRE( hydra::get<2>(table[i]) ==  Approx(hydra::get<2>(other[i])) );
			REQUIRE( hydra::get<2>(table[i]) ==  Approx(hydra::get<2>(other[i])) );
		}
	}

	SECTION( "copy constructor (different backend) multivector(other)" )
	{

		table_d table(10);

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

		for(size_t i =0; i< table.size(); i++ ){
			table[i] = hydra::make_tuple(i,i,i,i);
		}

		table_h other( table);

		REQUIRE( other.empty() == false );
		REQUIRE( other.size()  == 10 );
		REQUIRE( other.capacity() >= 10 );

		for(size_t i =0; i< table.size(); i++ ){
			REQUIRE( hydra::get<0>(table[i]) ==  (hydra::get<0>(other[i])) );
			REQUIRE( hydra::get<1>(table[i]) ==  (hydra::get<1>(other[i])) );
			REQUIRE( hydra::get<2>(table[i]) ==  Approx(hydra::get<2>(other[i])) );
			REQUIRE( hydra::get<2>(table[i]) ==  Approx(hydra::get<2>(other[i])) );
		}
	}

	SECTION( "copy constructor (different backend, iterators) multivector(other)" )
	{

		table_d table(10);

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

		for(size_t i =0; i< table.size(); i++ ){
			table[i] = hydra::make_tuple(i,i,i,i);
		}

		table_h other( table.begin(), table.end());

		REQUIRE( other.empty() == false );
		REQUIRE( other.size()  == 10 );
		REQUIRE( other.capacity() >= 10 );

		for(size_t i =0; i< table.size(); i++ ){
			REQUIRE( hydra::get<0>(table[i]) ==  (hydra::get<0>(other[i])) );
			REQUIRE( hydra::get<1>(table[i]) ==  (hydra::get<1>(other[i])) );
			REQUIRE( hydra::get<2>(table[i]) ==  Approx(hydra::get<2>(other[i])) );
			REQUIRE( hydra::get<2>(table[i]) ==  Approx(hydra::get<2>(other[i])) );
		}
	}

	SECTION( "move constructor multivector(other)" )
	{
		table_d table(10);

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );

		for(size_t i =0; i< table.size(); i++ ){
			table[i] = hydra::make_tuple(i,i,i,i);
		}

		table_d other( std::move(table));

		REQUIRE( other.empty() == false );
		REQUIRE( other.size()  == 10 );
		REQUIRE( other.capacity() >= 10 );

		for(size_t i =0; i< 10; i++ ){

			table.push_back(hydra::make_tuple(i+1,i+1,i+1,i+1));

		}

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 10 );
		REQUIRE( table.capacity() >= 10 );


		for(size_t i =0; i< 10; i++ ){

			REQUIRE( hydra::get<0>(table[i]) !=  hydra::get<0>(other[i]) );
			REQUIRE( hydra::get<1>(table[i]) !=  hydra::get<1>(other[i]) );
			REQUIRE( hydra::get<2>(table[i]) !=  Approx(hydra::get<2>(other[i])) );
			REQUIRE( hydra::get<3>(table[i]) !=  Approx(hydra::get<3>(other[i])) );
		}

		for(size_t i =0; i< 10; i++ ){

			REQUIRE( hydra::get<0>(table[i]) ==  (i+1)) ;
			REQUIRE( hydra::get<1>(table[i]) ==  (i+1)) ;
			REQUIRE( hydra::get<2>(table[i]) ==  Approx(i+1)) ;
			REQUIRE( hydra::get<3>(table[i]) ==  Approx(i+1)) ;

		}

		for(size_t i =0; i< 10; i++ ){

			REQUIRE( hydra::get<0>(other[i]) ==  (i)) ;
			REQUIRE( hydra::get<1>(other[i]) ==  (i)) ;
			REQUIRE( hydra::get<2>(other[i]) ==  Approx(i)) ;
			REQUIRE( hydra::get<3>(other[i]) ==  Approx(i)) ;
		}

	}

	SECTION( "assignment (same backend) multivector(other)" )
		{
			table_d table(10);

			REQUIRE( table.empty() == false );
			REQUIRE( table.size()  == 10 );
			REQUIRE( table.capacity() >= 10 );

			for(size_t i =0; i< table.size(); i++ ){
				table[i] = hydra::make_tuple(i,i,i,i);
			}

			table_d other= table;

			REQUIRE( other.empty() == false );
			REQUIRE( other.size()  == 10 );
			REQUIRE( other.capacity() >= 10 );

			for(size_t i =0; i< table.size(); i++ ){
				REQUIRE( hydra::get<0>(table[i]) ==  (hydra::get<0>(other[i])) );
				REQUIRE( hydra::get<1>(table[i]) ==  (hydra::get<1>(other[i])) );
				REQUIRE( hydra::get<2>(table[i]) ==  Approx(hydra::get<2>(other[i])) );
				REQUIRE( hydra::get<2>(table[i]) ==  Approx(hydra::get<2>(other[i])) );
			}
		}

	SECTION( "assignment (different backend) multivector(other)" )
		{

			table_d table(10);

			REQUIRE( table.empty() == false );
			REQUIRE( table.size()  == 10 );
			REQUIRE( table.capacity() >= 10 );

			for(size_t i =0; i< table.size(); i++ ){
				table[i] = hydra::make_tuple(i,i,i,i);
			}

			table_h other= table;

			REQUIRE( other.empty() == false );
			REQUIRE( other.size()  == 10 );
			REQUIRE( other.capacity() >= 10 );

			for(size_t i =0; i< table.size(); i++ ){
				REQUIRE( hydra::get<0>(table[i]) ==  (hydra::get<0>(other[i])) );
				REQUIRE( hydra::get<1>(table[i]) ==  (hydra::get<1>(other[i])) );
				REQUIRE( hydra::get<2>(table[i]) ==  Approx(hydra::get<2>(other[i])) );
				REQUIRE( hydra::get<2>(table[i]) ==  Approx(hydra::get<2>(other[i])) );
			}
		}

	SECTION( "assignment/move multivector(other)" )
		{
			table_d table(10);

			REQUIRE( table.empty() == false );
			REQUIRE( table.size()  == 10 );
			REQUIRE( table.capacity() >= 10 );

			for(size_t i =0; i< table.size(); i++ ){
				table[i] = hydra::make_tuple(i,i,i,i);
			}

			table_d other= std::move(table);

			REQUIRE( other.empty() == false );
			REQUIRE( other.size()  == 10 );
			REQUIRE( other.capacity() >= 10 );

			for(size_t i =0; i< 10; i++ ){

				table.push_back(hydra::make_tuple(i+1,i+1,i+1,i+1));

			}

			REQUIRE( table.empty() == false );
			REQUIRE( table.size()  == 10 );
			REQUIRE( table.capacity() >= 10 );


			for(size_t i =0; i< 10; i++ ){

				REQUIRE( hydra::get<0>(table[i]) !=  hydra::get<0>(other[i]) );
				REQUIRE( hydra::get<1>(table[i]) !=  hydra::get<1>(other[i]) );
				REQUIRE( hydra::get<2>(table[i]) !=  Approx(hydra::get<2>(other[i])) );
				REQUIRE( hydra::get<3>(table[i]) !=  Approx(hydra::get<3>(other[i])) );
			}

			for(size_t i =0; i< 10; i++ ){

				REQUIRE( hydra::get<0>(table[i]) ==  (i+1)) ;
				REQUIRE( hydra::get<1>(table[i]) ==  (i+1)) ;
				REQUIRE( hydra::get<2>(table[i]) ==  Approx(i+1)) ;
				REQUIRE( hydra::get<3>(table[i]) ==  Approx(i+1)) ;

			}

			for(size_t i =0; i< 10; i++ ){

				REQUIRE( hydra::get<0>(other[i]) ==  (i)) ;
				REQUIRE( hydra::get<1>(other[i]) ==  (i)) ;
				REQUIRE( hydra::get<2>(other[i]) ==  Approx(i)) ;
				REQUIRE( hydra::get<3>(other[i]) ==  Approx(i)) ;
			}

		}

	SECTION( "push_back" )
	{
			table_d table;

			REQUIRE( table.empty() == true );
			REQUIRE( table.size()  == 0 );
			REQUIRE( table.capacity() == 0 );

			for(size_t i =0; i< 10; i++ ){

				table.push_back(hydra::make_tuple(i,i,i,i));

			}

			REQUIRE( table.empty() == false );
			REQUIRE( table.size()  == 10 );
			REQUIRE( table.capacity() >= 10 );

			for(size_t i =0; i< 10; i++ ){
				REQUIRE( hydra::get<0>(table[i]) ==  i );
				REQUIRE( hydra::get<1>(table[i]) ==  i );
				REQUIRE( hydra::get<2>(table[i]) ==  Approx(i) );
				REQUIRE( hydra::get<3>(table[i]) ==  Approx(i) );
			}
	}

	SECTION( "pop_back" )
	{
			table_d table(1);

			REQUIRE( table.empty() == false );
			REQUIRE( table.size()  == 1 );
			REQUIRE( table.capacity() >= 1 );

			table.pop_back();

			REQUIRE( table.empty() == true );
			REQUIRE( table.size()  == 0 );
			REQUIRE( table.capacity() >= 0 );

	}

	SECTION( "front" )
	{
		table_d table(1, hydra::make_tuple(1,1,1,1));

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 1 );
		REQUIRE( table.capacity() >= 1 );

		table.front() = hydra::make_tuple(1,1,1,1);

		REQUIRE( hydra::get<0>(table[0]) ==  1 );
		REQUIRE( hydra::get<1>(table[0]) ==  1 );
		REQUIRE( hydra::get<2>(table[0]) ==  Approx(1) );
		REQUIRE( hydra::get<3>(table[0]) ==  Approx(1) );


	}

	SECTION( "back" )
	{
		table_d table(1, hydra::make_tuple(1,1,1,1));

		REQUIRE( table.empty() == false );
		REQUIRE( table.size()  == 1 );
		REQUIRE( table.capacity() >= 1 );

		table.back() = hydra::make_tuple(1,1,1,1);

		REQUIRE( hydra::get<0>(table[0]) ==  1 );
		REQUIRE( hydra::get<1>(table[0]) ==  1 );
		REQUIRE( hydra::get<2>(table[0]) ==  Approx(1) );
		REQUIRE( hydra::get<3>(table[0]) ==  Approx(1) );

	}


}

