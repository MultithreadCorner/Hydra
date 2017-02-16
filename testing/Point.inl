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

#include <array>


TEST_CASE( "Point","hydra::experimental::Point" ) {

	typedef  hydra::experimental::Point<double, 3> point1;

	typedef  hydra::experimental::Point<double, 3, true> point2;

	typedef  hydra::experimental::Point<double, 3, true, true> point3;


	SECTION( "Constructors " )
	{

		SECTION( "default constructor" )
		{

			point1 point;

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(0.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(0.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(0.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(0.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(0.0) );
	    }

		SECTION( "trivial constructor" )
		{
			point1 point( typename point1::type(2.0, 4.0, 1.0, 2.0, 3.0) );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(3.0) );

		}

		SECTION( "constructor from std::array" )
		{
			std::array<double, 3> array{1.0, 2.0, 3.0};
			point1 point( array , 2.0 );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(3.0) );

		}

		SECTION( "constructor from static array" )
		{
			double array[3] = {1.0, 2.0, 3.0};
			point1 point( array, 2.0 );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(3.0) );

		}

		SECTION( "constructor from std::initializer_list" )
		{
			point1 point( {1.0, 2.0, 3.0} , 2.0 );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(3.0) );

		}

		SECTION( "constructor from  coordinate_type" )
		{
			point1 point( typename point1::coordinate_type(1.0, 2.0, 3.0) , 2.0 );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(3.0) );

		}


		SECTION( "constructor from pointer array" )
		{
			double array[3] = {1.0, 2.0, 3.0};
			point1 point( &array[0], 2.0 );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(3.0) );

		}

		SECTION( "copy constructor" )
		{
			double array[3] = {1.0, 2.0, 3.0};
			point1 pointA( &array[0], 2.0 );
			point1 pointB( pointA);


			auto data = pointB.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(3.0) );

		}



	}

	SECTION( "Operators" )
	{

		SECTION( "assignment to another point" )
		{
			double arrayA[3] = {1.0, 2.0, 3.0};
			double arrayB[3] = {2.0, 4.0, 6.0};

			point1 pointA( &arrayA[0], 2.0 );
			point1 pointB( &arrayB[0], 4.0 );

			pointB = pointA;

			auto data = pointB.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(3.0) );

		}

		SECTION( "assignment to tuple" )
		{
			double arrayA[3] = {1.0, 2.0, 3.0};
			auto tpl = thrust::make_tuple(2.0, 4.0, 2.0, 4.0, 6.0);

			point1 pointA( &arrayA[0], 2.0 );

			pointA = pointA;

			auto data = pointA.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(3.0) );

		}

		SECTION( "conversion to tuple" )
		{
			double arrayA[3] = {1.0, 2.0, 3.0};
			auto tpl = thrust::make_tuple(2.0, 4.0, 2.0, 4.0, 6.0);

			point1 pointA( &arrayA[0], 2.0 );

		    tpl = pointA;

			auto data = pointA.GetData();

			REQUIRE( thrust::get<0>(tpl) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(tpl) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(tpl) ==  Approx(1.0) );
			REQUIRE( thrust::get<3>(tpl) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(tpl) ==  Approx(3.0) );

		}


	}


	SECTION( "Setters and getters" )
	{
		SECTION( "getters" )
		{
			double array[3] = {1.0, 2.0, 3.0};
			point1 point( &array[0], 2.0 );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(3.0) );

			auto coordinates =  point.GetCoordinates();

			REQUIRE( point.GetWeight()  ==  Approx(2.0) );
			REQUIRE( point.GetWeight2() ==  Approx(4.0) );
			REQUIRE( thrust::get<0>(coordinates) ==  Approx(1.0) );
			REQUIRE( thrust::get<1>(coordinates) ==  Approx(2.0) );
			REQUIRE( thrust::get<2>(coordinates) ==  Approx(3.0) );

		}

		SECTION( "setters" )
		{
			double array[3] = {1.0, 2.0, 3.0};
			point1 point;

			point.SetData( thrust::make_tuple( 2.0, 4.0, 1.0, 2.0, 3.0 ) );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(3.0) );

			point.SetCoordinate( thrust::make_tuple( 2.0, 4.0, 6.0) );
			point.SetWeight( 3.0 )  ;
			point.SetWeight2( 9.0 ) ;

			data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(3.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(9.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(6.0) );

		}

	}

}
