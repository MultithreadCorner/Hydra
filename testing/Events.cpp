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
 * events_testing.cpp
 *
 *  Created on: 29/10/2016
 *      Author: Antonio Augusto Alves Junior
 */

#include <memory>
#include <limits>
#include <utility>
#include "catch/catch.hpp"
#include <hydra/experimental/Chain.h>
#include <hydra/experimental/Events.h>
#include <hydra/experimental/multivector.h>
#include <hydra/experimental/Vector4R.h>

using namespace std;
using namespace hydra;

TEST_CASE( "Events","hydra::Events" ) {

	typedef  experimental::Events<3,  host> events3_t;

	SECTION( "move semantics: constructor" )
	{
		events3_t events(10);
		REQUIRE( events.GetNEvents()  == 10 );

		size_t i = 0;
		for(auto ev:events )
		{
			thrust::get<0>(ev) = i;
			thrust::get<1>(ev) = (typename experimental::Vector4R::args_type) experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>(ev) = (typename experimental::Vector4R::args_type) experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>(ev) = (typename experimental::Vector4R::args_type) experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
			i++;
		}

		events3_t other( std::move(events) );


		events.resize(10);

		i = 1;
		for(auto ev:events )
		{
			thrust::get<0>(ev) = i;
			thrust::get<1>(ev) = (typename experimental::Vector4R::args_type) experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>(ev) = (typename experimental::Vector4R::args_type) experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>(ev) = (typename experimental::Vector4R::args_type) experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
			i++;
		}

		i = 0;
		for(auto ev:other )
		{
			REQUIRE( thrust::get<0>(ev) == i );
			REQUIRE( thrust::get<1>(ev) == thrust::make_tuple(1+i, 1+i, 1+i, 1+i) );
			REQUIRE( thrust::get<2>(ev) == thrust::make_tuple(2+i, 2+i, 2+i, 2+i) );
			REQUIRE( thrust::get<3>(ev) == thrust::make_tuple(3+i, 3+i, 3+i, 3+i) );
			i++;
		}

	}


	SECTION( "move semantics: assignment" )
	{
		events3_t events(10);
		REQUIRE( events.GetNEvents()  == 10 );

		size_t i = 0;
		for(auto ev:events )
		{
			thrust::get<0>(ev) = i;
			thrust::get<1>(ev) = (typename experimental::Vector4R::args_type) experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>(ev) = (typename experimental::Vector4R::args_type) experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>(ev) = (typename experimental::Vector4R::args_type) experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
			i++;
		}

		events3_t other = std::move(events) ;

		REQUIRE( events.capacity()==0);


		events.resize(10);
		i = 1;
		for(auto ev:events )
		{
			thrust::get<0>(ev) = i;
			thrust::get<1>(ev) = (typename experimental::Vector4R::args_type) experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>(ev) = (typename experimental::Vector4R::args_type) experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>(ev) = (typename experimental::Vector4R::args_type) experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
			i++;
		}

		i = 0;
		for(auto ev:other )
		{
			REQUIRE( thrust::get<0>(ev) == i );
			REQUIRE( thrust::get<1>(ev) == thrust::make_tuple(1+i, 1+i, 1+i, 1+i) );
			REQUIRE( thrust::get<2>(ev) == thrust::make_tuple(2+i, 2+i, 2+i, 2+i) );
			REQUIRE( thrust::get<3>(ev) == thrust::make_tuple(3+i, 3+i, 3+i, 3+i) );
			i++;
		}

	}


	SECTION( "conversion Vector4R -> tuple at iterator level access" )
	{
		events3_t events(10);
		REQUIRE( events.GetNEvents()  == 10 );

		size_t i = 0;
		for(auto ev:events )
		{
			thrust::get<0>(ev) = i;
			thrust::get<1>(ev) = (typename experimental::Vector4R::args_type) experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>(ev) = (typename experimental::Vector4R::args_type) experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>(ev) = (typename experimental::Vector4R::args_type) experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
			i++;
		}

		i = 0;
		for(auto ev:events )
		{
			REQUIRE( thrust::get<0>(ev) == i );
			REQUIRE( thrust::get<1>(ev) == thrust::make_tuple(1+i, 1+i, 1+i, 1+i) );
			REQUIRE( thrust::get<2>(ev) == thrust::make_tuple(2+i, 2+i, 2+i, 2+i) );
			REQUIRE( thrust::get<3>(ev) == thrust::make_tuple(3+i, 3+i, 3+i, 3+i) );
			i++;
		}

	}


	SECTION( "conversion Vector4R -> tuple via subscript operator access " )
	{
		events3_t events(10);
		REQUIRE( events.GetNEvents()  == 10 );

		size_t i = 0;
		for( size_t i = 0; i< events.GetNEvents(); i++ )
		{
			thrust::get<0>( events[i]) = i;
			thrust::get<1>( events[i]) = (typename experimental::Vector4R::args_type) experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>( events[i]) = (typename experimental::Vector4R::args_type) experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>( events[i]) = (typename experimental::Vector4R::args_type) experimental::Vector4R(3+i, 3+i, 3+i, 3+i);

		}


		for( size_t i = 0; i< events.GetNEvents(); i++ )
		{
			REQUIRE( thrust::get<0>(events[i]) == i );
			REQUIRE( thrust::get<1>(events[i]) == thrust::make_tuple(1+i, 1+i, 1+i, 1+i) );
			REQUIRE( thrust::get<2>(events[i]) == thrust::make_tuple(2+i, 2+i, 2+i, 2+i) );
			REQUIRE( thrust::get<3>(events[i]) == thrust::make_tuple(3+i, 3+i, 3+i, 3+i) );
		}
	}


	SECTION( "conversion tuple -> Vector4R via subscript operator access" )
	{
		events3_t events(10);
		REQUIRE( events.GetNEvents()  == 10 );

		size_t i = 0;
		for( size_t i = 0; i< events.GetNEvents(); i++ )
		{
			thrust::get<0>( events[i]) = i;
			thrust::get<1>( events[i]) = thrust::make_tuple(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>( events[i]) = thrust::make_tuple(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>( events[i]) = thrust::make_tuple(3+i, 3+i, 3+i, 3+i);
		}

		for( size_t i = 0; i< events.GetNEvents(); i++ )
		{
			experimental::Vector4R v1=	thrust::get<1>( events[i]);
			experimental::Vector4R v2=	thrust::get<2>( events[i]);
			experimental::Vector4R v3=	thrust::get<3>( events[i]);

			REQUIRE( v1.get(0) == 1+i );
			REQUIRE( v1.get(1) == 1+i );
			REQUIRE( v1.get(2) == 1+i );
			REQUIRE( v1.get(3) == 1+i );

			REQUIRE( v2.get(0) == 2+i );
			REQUIRE( v2.get(1) == 2+i );
			REQUIRE( v2.get(2) == 2+i );
			REQUIRE( v2.get(3) == 2+i );

			REQUIRE( v3.get(0) == 3+i );
			REQUIRE( v3.get(1) == 3+i );
			REQUIRE( v3.get(2) == 3+i );
			REQUIRE( v3.get(3) == 3+i );
		}

	}


}
