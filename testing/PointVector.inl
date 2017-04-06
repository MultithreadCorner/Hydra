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
 * PointVector.inl
 *
 *  Created on: 06/04/2017
 *      Author: Antonio Augusto Alves Junior
 */

#pragma once

#include <memory>
#include <limits>
#include <utility>

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/experimental/Point.h>
#include <hydra/experimental/PointVector.h>
#include <thrust/tuple.h>


#include <array>
TEST_CASE( "PointVector","hydra::experimental::PointVector 1D case" ) {

	typedef  hydra::experimental::Point<double, 3> Point_t;
	typedef  hydra::experimental::PointVector<Point_t, hydra::host> PointVector_h;
	typedef  hydra::experimental::PointVector<Point_t, hydra::device> PointVector_d;

	SECTION( "Constructors " )
	{
		SECTION( "default constructor" )
				{
			auto points = PointVector_h();

			REQUIRE( points.size() ==  0.0 );


				}

	}

}
