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
 * Point.inl
 *
 *  Created on: 16/02/2017
 *      Author: Antonio Augusto Alves Junior
 */

#pragma once

#include <memory>
#include <limits>
#include <utility>

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/experimental/Point.h>
#include <thrust/tuple.h>



TEST_CASE( "Point","hydra::experimental::Point" ) {

	typedef  hydra::experimental::Point<double, 3> point1;

	typedef  hydra::experimental::Point<double, 3, true> point2;

	typedef  hydra::experimental::Point<double, 3, true, true> point3;


	SECTION( "default constructor, " )
	{
		point1 point;

		auto data = point.GetData();

		REQUIRE( point.GetWeight()    ==  Approx(0.0) );
		REQUIRE( point.GetWeight2()   ==  Approx(0.0) );
		REQUIRE( thrust::get<0>(data) ==  Approx(0.0) );
		REQUIRE( thrust::get<1>(data) ==  Approx(0.0) );
		REQUIRE( thrust::get<2>(data) ==  Approx(0.0) );


	}


}
