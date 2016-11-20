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


TEST_CASE( "Chain","hydra::Chain" ) {

	typedef  hydra::experimental::Events<3,  hydra::host> events3_t;
	typedef  hydra::experimental::Events<2,  hydra::host> events2_t;
	typedef  hydra::experimental::Chain<events3_t, events2_t> chain_t;

	SECTION( "constructor Chain(n)" )
	{
		chain_t events(10);
		REQUIRE( events.GetNEvents()  == 10 );


		for( size_t i = 0; i< events.GetNEvents(); i++ )
		{
			auto weight = thrust::get<0>(events[i]);
			auto decay1 = thrust::get<1>(events[i]);
			auto decay2 = thrust::get<2>(events[i]);

			thrust::get<0>( decay1) = 1;
			thrust::get<1>( decay1) = thrust::make_tuple(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>( decay1) = thrust::make_tuple(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>( decay1) = thrust::make_tuple(3+i, 3+i, 3+i, 3+i);

			thrust::get<0>( decay2) = 2;
			thrust::get<1>( decay2) = thrust::make_tuple(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>( decay2) = thrust::make_tuple(2+i, 2+i, 2+i, 2+i);

		}


		for( size_t i = 0; i< events.GetNEvents(); i++ )
		{
			auto weight = thrust::get<0>(events[i]);
			auto decay1 = thrust::get<1>(events[i]);
			auto decay2 = thrust::get<2>(events[i]);


			REQUIRE( thrust::get<0>( decay1) == 1);
			REQUIRE( thrust::get<1>( decay1) == thrust::make_tuple(1+i, 1+i, 1+i, 1+i));
			REQUIRE( thrust::get<2>( decay1) == thrust::make_tuple(2+i, 2+i, 2+i, 2+i));
			REQUIRE( thrust::get<3>( decay1) == thrust::make_tuple(3+i, 3+i, 3+i, 3+i));

			REQUIRE( thrust::get<0>( decay2) == 2);
			REQUIRE( thrust::get<1>( decay2) == thrust::make_tuple(1+i, 1+i, 1+i, 1+i));
			REQUIRE( thrust::get<2>( decay2) == thrust::make_tuple(2+i, 2+i, 2+i, 2+i));
		}

	}


	SECTION( "constructor Chain(events...)" )
	{
		events3_t events3(10);
		events2_t events2(10);

		size_t i = 0;
		for( size_t i = 0; i< events3.GetNEvents(); i++ )
		{
			thrust::get<0>( events3[i]) = i;
			thrust::get<1>( events3[i]) = thrust::make_tuple(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>( events3[i]) = thrust::make_tuple(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>( events3[i]) = thrust::make_tuple(3+i, 3+i, 3+i, 3+i);

			thrust::get<0>( events2[i]) = i;
			thrust::get<1>( events2[i]) = thrust::make_tuple(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>( events2[i]) = thrust::make_tuple(2+i, 2+i, 2+i, 2+i);

		}

		REQUIRE( events3.capacity()==10);
		REQUIRE( events2.capacity()==10);


		chain_t chain(events3, events2);

		REQUIRE( events3.capacity()==0);
		REQUIRE( events2.capacity()==0);


		i = 0;
		for( size_t i = 0; i< chain.GetNEvents(); i++ )
		{
			auto weight = thrust::get<0>(chain[i]);
			auto decay1 = thrust::get<1>(chain[i]);
			auto decay2 = thrust::get<2>(chain[i]);


			REQUIRE( thrust::get<0>( decay1) == i);
			REQUIRE( thrust::get<1>( decay1) == thrust::make_tuple(1+i, 1+i, 1+i, 1+i));
			REQUIRE( thrust::get<2>( decay1) == thrust::make_tuple(2+i, 2+i, 2+i, 2+i));
			REQUIRE( thrust::get<3>( decay1) == thrust::make_tuple(3+i, 3+i, 3+i, 3+i));

			REQUIRE( thrust::get<0>( decay2) == i);
			REQUIRE( thrust::get<1>( decay2) == thrust::make_tuple(1+i, 1+i, 1+i, 1+i));
			REQUIRE( thrust::get<2>( decay2) == thrust::make_tuple(2+i, 2+i, 2+i, 2+i));

		}
	}

	SECTION( "move constructor Chain(other)" )
	{
		chain_t events(10);
		REQUIRE( events.capacity()  == 10 );

		size_t i = 0;
		for( size_t i = 0; i< events.GetNEvents(); i++ )
		{
			auto weight = thrust::get<0>(events[i]);
			auto decay1 = thrust::get<1>(events[i]);
			auto decay2 = thrust::get<2>(events[i]);

			thrust::get<0>( decay1) = 1;
			thrust::get<1>( decay1) = thrust::make_tuple(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>( decay1) = thrust::make_tuple(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>( decay1) = thrust::make_tuple(3+i, 3+i, 3+i, 3+i);

			thrust::get<0>( decay2) = 2;
			thrust::get<1>( decay2) = thrust::make_tuple(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>( decay2) = thrust::make_tuple(2+i, 2+i, 2+i, 2+i);
		}


		chain_t events2(std::move(events));
		REQUIRE( events.GetNEvents()  == 0 );
		i = 0;
		for( size_t i = 0; i< events2.GetNEvents(); i++ )
		{
			auto weight = thrust::get<0>(events2[i]);
			auto decay1 = thrust::get<1>(events2[i]);
			auto decay2 = thrust::get<2>(events2[i]);


			REQUIRE( thrust::get<0>( decay1) == 1);
			REQUIRE( thrust::get<1>( decay1) == thrust::make_tuple(1+i, 1+i, 1+i, 1+i));
			REQUIRE( thrust::get<2>( decay1) == thrust::make_tuple(2+i, 2+i, 2+i, 2+i));
			REQUIRE( thrust::get<3>( decay1) == thrust::make_tuple(3+i, 3+i, 3+i, 3+i));

			REQUIRE( thrust::get<0>( decay2) == 2);
			REQUIRE( thrust::get<1>( decay2) == thrust::make_tuple(1+i, 1+i, 1+i, 1+i));
			REQUIRE( thrust::get<2>( decay2) == thrust::make_tuple(2+i, 2+i, 2+i, 2+i));

		}

	}

	SECTION( "move assingment" )
	{
		chain_t events(10);
		REQUIRE( events.capacity()  == 10 );

		size_t i = 0;
		for( size_t i = 0; i< events.GetNEvents(); i++ )
		{
			auto weight = thrust::get<0>(events[i]);
			auto decay1 = thrust::get<1>(events[i]);
			auto decay2 = thrust::get<2>(events[i]);

			thrust::get<0>( decay1) = 1;
			thrust::get<1>( decay1) = thrust::make_tuple(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>( decay1) = thrust::make_tuple(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>( decay1) = thrust::make_tuple(3+i, 3+i, 3+i, 3+i);

			thrust::get<0>( decay2) = 2;
			thrust::get<1>( decay2) = thrust::make_tuple(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>( decay2) = thrust::make_tuple(2+i, 2+i, 2+i, 2+i);
		}


		chain_t events2 = std::move(events);
		REQUIRE( events.GetNEvents()  == 0 );
		i = 0;
		for( size_t i = 0; i< events2.GetNEvents(); i++ )
		{
			auto weight = thrust::get<0>(events2[i]);
			auto decay1 = thrust::get<1>(events2[i]);
			auto decay2 = thrust::get<2>(events2[i]);


			REQUIRE( thrust::get<0>( decay1) == 1);
			REQUIRE( thrust::get<1>( decay1) == thrust::make_tuple(1+i, 1+i, 1+i, 1+i));
			REQUIRE( thrust::get<2>( decay1) == thrust::make_tuple(2+i, 2+i, 2+i, 2+i));
			REQUIRE( thrust::get<3>( decay1) == thrust::make_tuple(3+i, 3+i, 3+i, 3+i));

			REQUIRE( thrust::get<0>( decay2) == 2);
			REQUIRE( thrust::get<1>( decay2) == thrust::make_tuple(1+i, 1+i, 1+i, 1+i));
			REQUIRE( thrust::get<2>( decay2) == thrust::make_tuple(2+i, 2+i, 2+i, 2+i));

		}

	}

	SECTION( "copy constructor Chain(other)" )
	{
		chain_t events(10);
		REQUIRE( events.capacity()  == 10 );

		size_t i = 0;
		for( size_t i = 0; i< events.GetNEvents(); i++ )
		{
			auto weight = thrust::get<0>(events[i]);
			auto decay1 = thrust::get<1>(events[i]);
			auto decay2 = thrust::get<2>(events[i]);

			thrust::get<0>( decay1) = 1;
			thrust::get<1>( decay1) = thrust::make_tuple(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>( decay1) = thrust::make_tuple(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>( decay1) = thrust::make_tuple(3+i, 3+i, 3+i, 3+i);

			thrust::get<0>( decay2) = 2;
			thrust::get<1>( decay2) = thrust::make_tuple(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>( decay2) = thrust::make_tuple(2+i, 2+i, 2+i, 2+i);
		}


		chain_t events2(events);
		REQUIRE( events.GetNEvents()  == 10 );

		i = 0;
		for( size_t i = 0; i< events2.GetNEvents(); i++ )
		{
			auto weight = thrust::get<0>(events2[i]);
			auto A1 = thrust::get<1>(events2[i]);
			auto A2 = thrust::get<2>(events2[i]);
			auto B1 = thrust::get<1>(events[i]);
			auto B2 = thrust::get<2>(events[i]);


			REQUIRE( thrust::get<0>( A1) == thrust::get<0>( B1));
			REQUIRE( thrust::get<1>( A1) == thrust::get<1>( B1));
			REQUIRE( thrust::get<2>( A1) == thrust::get<2>( B1));
			REQUIRE( thrust::get<3>( A1) == thrust::get<3>( B1));

			REQUIRE( thrust::get<0>( A2) == thrust::get<0>( B2));
			REQUIRE( thrust::get<1>( A2) == thrust::get<1>( B2));
			REQUIRE( thrust::get<2>( A2) == thrust::get<2>( B2));

		}

	}


	SECTION( "copy assignment" )
	{
		chain_t events(10);
		REQUIRE( events.capacity()  == 10 );

		size_t i = 0;
		for( size_t i = 0; i< events.GetNEvents(); i++ )
		{
			auto weight = thrust::get<0>(events[i]);
			auto decay1 = thrust::get<1>(events[i]);
			auto decay2 = thrust::get<2>(events[i]);

			thrust::get<0>( decay1) = 1;
			thrust::get<1>( decay1) = thrust::make_tuple(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>( decay1) = thrust::make_tuple(2+i, 2+i, 2+i, 2+i);
			thrust::get<3>( decay1) = thrust::make_tuple(3+i, 3+i, 3+i, 3+i);

			thrust::get<0>( decay2) = 2;
			thrust::get<1>( decay2) = thrust::make_tuple(1+i, 1+i, 1+i, 1+i);
			thrust::get<2>( decay2) = thrust::make_tuple(2+i, 2+i, 2+i, 2+i);
		}


		chain_t events2=events;
		REQUIRE( events.GetNEvents()  == 10 );

		i = 0;
		for( size_t i = 0; i< events2.GetNEvents(); i++ )
		{
			auto weight = thrust::get<0>(events2[i]);
			auto A1 = thrust::get<1>(events2[i]);
			auto A2 = thrust::get<2>(events2[i]);
			auto B1 = thrust::get<1>(events[i]);
			auto B2 = thrust::get<2>(events[i]);


			REQUIRE( thrust::get<0>( B1) == 1);
			REQUIRE( thrust::get<1>( B1) == thrust::make_tuple(1+i, 1+i, 1+i, 1+i));
			REQUIRE( thrust::get<2>( B1) == thrust::make_tuple(2+i, 2+i, 2+i, 2+i) );
			REQUIRE( thrust::get<3>( B1) == thrust::make_tuple(3+i, 3+i, 3+i, 3+i));

			REQUIRE( thrust::get<0>( B2) == 2);
			REQUIRE( thrust::get<1>( B2) == thrust::make_tuple(1+i, 1+i, 1+i, 1+i));
			REQUIRE( thrust::get<2>( B2) == thrust::make_tuple(2+i, 2+i, 2+i, 2+i));

		}

	}

}
