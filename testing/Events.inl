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
#pragma once

#include <memory>
#include <limits>
#include <utility>

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/experimental/Chain.h>
#include <hydra/experimental/Events.h>
#include <hydra/experimental/multivector.h>
#include <hydra/experimental/Vector4R.h>


TEST_CASE( "Events","hydra::Events" ) {

	typedef  hydra::experimental::Events<3, hydra::host> events3_h;
	typedef  hydra::experimental::Events<3, hydra::device> events3_d;

	SECTION( "copy constructor <device backend>" )
	{
		events3_d events(10);
		REQUIRE( events.GetNEvents()  == 10 );

		size_t i = 0;
		for(auto ev:events )
		{
			thrust::get<0>(ev) = i;
			thrust::get<1>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
			i++;
		}

		events3_d other( events );

		//test containers have same number of elements
		REQUIRE( events.GetNEvents()  ==  other.GetNEvents() );

		//test elements are the same
		for( size_t i=0; i <  events.GetNEvents(); i++)
		{
			auto ev1 = events[i];
			auto ev2 = other[i];
			REQUIRE( ev1  ==  ev2 );
		}

		//te
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



	SECTION( "copy constructor <host backend>" )
		{
			events3_h events(10);
			REQUIRE( events.GetNEvents()  == 10 );

			size_t i = 0;
			for(auto ev:events )
			{
				thrust::get<0>(ev) = i;
				thrust::get<1>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
				thrust::get<2>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
				thrust::get<3>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
				i++;
			}

			events3_h other( events );

			//test containers have same number of elements
			REQUIRE( events.GetNEvents()  ==  other.GetNEvents() );

			//test elements are the same
			for( size_t i=0; i <  events.GetNEvents(); i++)
			{
				auto ev1 = events[i];
				auto ev2 = other[i];
				REQUIRE( ev1  ==  ev2 );
			}

			//te
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


	SECTION( "move assignment <device backend>" )
	{
		events3_d events(10);
		REQUIRE( events.GetNEvents()  == 10 );

		size_t i = 0;
		for(auto ev:events )
		{
			thrust::get<0>(ev) = i;
			thrust::get<1>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
			i++;
		}

		events3_d other = std::move(events) ;

		REQUIRE( events.capacity()==0);


		events.resize(10);
		i = 1;
		for(auto ev:events )
		{
			thrust::get<0>(ev) = i;
			thrust::get<1>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
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


	SECTION( "move assignment <host backend>" )
		{
			events3_h events(10);

			REQUIRE( events.GetNEvents()  == 10 );

			size_t i = 0;
			for(auto ev:events )
			{
				thrust::get<0>(ev) = i;
				thrust::get<1>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
				thrust::get<2>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
				thrust::get<3>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
				i++;
			}

			events3_h other = std::move(events) ;

			REQUIRE( events.capacity()==0);


			events.resize(10);
			i = 1;
			for(auto ev:events )
			{
				thrust::get<0>(ev) = i;
				thrust::get<1>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
				thrust::get<2>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
				thrust::get<3>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
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

	SECTION( "move constructor <device backend>" )
	{
		events3_d events(10);
		REQUIRE( events.GetNEvents()  == 10 );

		size_t i = 0;
		for(auto ev:events )
		{
			thrust::get<0>(ev) = i;
			thrust::get<1>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
			i++;
		}

		events3_d other( std::move(events) );

		REQUIRE( events.capacity()==0);

		events.resize(10);

		i = 1;
		for(auto ev:events )
		{
			thrust::get<0>(ev) = i;
			thrust::get<1>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
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


	SECTION( "move assignmen <host backend>t" )
	{
		events3_h events(10);
		REQUIRE( events.GetNEvents()  == 10 );

		size_t i = 0;
		for(auto ev:events )
		{
			thrust::get<0>(ev) = i;
			thrust::get<1>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
			i++;
		}

		events3_h other = std::move(events) ;

		REQUIRE( events.capacity()==0);

		events.resize(10);
		i = 1;
		for(auto ev:events )
		{
			thrust::get<0>(ev) = i;
			thrust::get<1>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
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
		events3_d events(10);
		REQUIRE( events.GetNEvents()  == 10 );

		size_t i = 0;
		for(auto ev:events )
		{
			thrust::get<0>(ev) = i;
			thrust::get<1>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>(ev) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(3+i, 3+i, 3+i, 3+i);
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
		events3_d events(10);
		REQUIRE( events.GetNEvents()  == 10 );

		size_t i = 0;
		for( size_t i = 0; i< events.GetNEvents(); i++ )
		{
			thrust::get<0>( events[i]) = i;
			thrust::get<1>( events[i]) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>( events[i]) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>( events[i]) = (typename hydra::experimental::Vector4R::args_type) hydra::experimental::Vector4R(3+i, 3+i, 3+i, 3+i);

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
		events3_d events(10);
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
			hydra::experimental::Vector4R v1=	thrust::get<1>( events[i]);
			hydra::experimental::Vector4R v2=	thrust::get<2>( events[i]);
			hydra::experimental::Vector4R v3=	thrust::get<3>( events[i]);

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
