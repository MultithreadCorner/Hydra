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


TEST_CASE( "Point<double, 3>","hydra::experimental::Point without value error" ) {

	typedef  hydra::experimental::Point<double, 3> Point_t;


	SECTION( "Constructors " )
	{

		SECTION( "default constructor" )
		{

			Point_t point;

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(0.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(0.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(0.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(0.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(0.0) );
	    }

		SECTION( "trivial constructor" )
		{
			Point_t point( typename Point_t::type(2.0, 4.0, 1.0, 2.0, 3.0) );

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
			Point_t point( array , 2.0 );

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
			Point_t point( array, 2.0 );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(3.0) );

		}

		SECTION( "constructor from std::initializer_list" )
		{
			Point_t point( {1.0, 2.0, 3.0} , 2.0 );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(3.0) );

		}

		SECTION( "constructor from  coordinate_type" )
		{
			Point_t point( typename Point_t::coordinate_type(1.0, 2.0, 3.0) , 2.0 );

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
			Point_t point( &array[0], 2.0 );

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
			Point_t pointA( &array[0], 2.0 );
			Point_t pointB( pointA);


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

			Point_t pointA( &arrayA[0], 2.0 );
			Point_t pointB( &arrayB[0], 4.0 );

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

			Point_t pointA( &arrayA[0], 2.0 );

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

			Point_t pointA( &arrayA[0], 2.0 );

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
			Point_t point( &array[0], 2.0 );

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
			Point_t point;

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



TEST_CASE( "Point<double, 3, true>","hydra::experimental::Point with value error" ) {

	typedef  hydra::experimental::Point<double, 3, true> Point_t;


	SECTION( "Constructors " )
	{

		SECTION( "default constructor" )
		{

			Point_t point;

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(0.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(0.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(0.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(0.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(0.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(0.0) );

	    }

		SECTION( "trivial constructor" )
		{
			Point_t point( typename Point_t::type(2.0, 4.0, 1.5, 1.0, 2.0, 3.0) );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.5) );
			REQUIRE( thrust::get<3>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<5>(data) ==  Approx(3.0) );

		}

		SECTION( "constructor from std::array" )
		{
			std::array<double, 3> array{1.0, 2.0, 3.0};
			Point_t point( array, 1.5 , 2.0 );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.5) );
			REQUIRE( thrust::get<3>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<5>(data) ==  Approx(3.0) );

		}

		SECTION( "constructor from static array" )
		{
			double array[3] = {1.0, 2.0, 3.0};
			Point_t point( array,1.5, 2.0 );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.5) );
			REQUIRE( thrust::get<3>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<5>(data) ==  Approx(3.0) );

		}

		SECTION( "constructor from std::initializer_list" )
		{
			Point_t point( {1.0, 2.0, 3.0} , 1.5, 2.0 );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.5) );
			REQUIRE( thrust::get<3>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<5>(data) ==  Approx(3.0) );


		}

		SECTION( "constructor from  coordinate_type" )
		{
			Point_t point( typename Point_t::coordinate_type(1.0, 2.0, 3.0), 1.5, 2.0 );

			auto data = point.GetData();


			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.5) );
			REQUIRE( thrust::get<3>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<5>(data) ==  Approx(3.0) );

		}


		SECTION( "constructor from pointer array" )
		{
			double array[3] = {1.0, 2.0, 3.0};
			Point_t point( &array[0], 1.5, 2.0 );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.5) );
			REQUIRE( thrust::get<3>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<5>(data) ==  Approx(3.0) );

		}

		SECTION( "copy constructor" )
		{
			double array[3] = {1.0, 2.0, 3.0};
			Point_t pointA( &array[0], 1.5, 2.0 );
			Point_t pointB( pointA);


			auto data = pointB.GetData();


			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.5) );
			REQUIRE( thrust::get<3>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<5>(data) ==  Approx(3.0) );

		}



	}

	SECTION( "Operators" )
	{

		SECTION( "assignment to another point" )
		{
			double arrayA[3] = {1.0, 2.0, 3.0};
			double arrayB[3] = {2.0, 4.0, 6.0};

			Point_t pointA( &arrayA[0], 1.5, 2.0 );
			Point_t pointB( &arrayB[0], 3.5, 4.0 );

			pointB = pointA;

			auto data = pointB.GetData();


			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.5) );
			REQUIRE( thrust::get<3>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<5>(data) ==  Approx(3.0) );

		}

		SECTION( "assignment to tuple" )
		{
			double arrayA[3] = {1.0, 2.0, 3.0};
			auto tpl = thrust::make_tuple(2.0, 4.0, 1.5, 2.0, 4.0, 6.0);

			Point_t pointA( &arrayA[0], 2.5, 2.0 );

			pointA = tpl;

			auto data = pointA.GetData();


			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.5) );
			REQUIRE( thrust::get<3>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<5>(data) ==  Approx(6.0) );

		}

		SECTION( "conversion to tuple" )
		{
			double arrayA[3] = {1.0, 2.0, 3.0};
			auto tpl = thrust::make_tuple(2.0, 4.0, 3.0, 2.0, 4.0, 6.0);

			Point_t pointA( &arrayA[0], 3.0, 2.0 );

		    tpl = pointA;

			auto data = pointA.GetData();


			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(3.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<5>(data) ==  Approx(3.0) );

		}


	}


	SECTION( "Setters and getters" )
	{
		SECTION( "getters" )
		{
			double array[3] = {1.0, 2.0, 3.0};
			Point_t point( &array[0], 1.5, 2.0 );

			auto data = point.GetData();


			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.5) );
			REQUIRE( thrust::get<3>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<5>(data) ==  Approx(3.0) );

			auto coordinates =  point.GetCoordinates();

			REQUIRE( point.GetWeight()  ==  Approx(2.0) );
			REQUIRE( point.GetWeight2() ==  Approx(4.0) );
			REQUIRE( point.GetError()   ==  Approx(1.5) );
			REQUIRE( thrust::get<0>(coordinates) ==  Approx(1.0) );
			REQUIRE( thrust::get<1>(coordinates) ==  Approx(2.0) );
			REQUIRE( thrust::get<2>(coordinates) ==  Approx(3.0) );

		}

		SECTION( "setters" )
		{
			double array[3] = {1.0, 2.0, 3.0};
			Point_t point;

			point.SetData( thrust::make_tuple( 2.0, 4.0, 1.5, 1.0, 2.0, 3.0 ) );

			auto data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(1.5) );
			REQUIRE( thrust::get<3>(data) ==  Approx(1.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<5>(data) ==  Approx(3.0) );

			point.SetCoordinate( thrust::make_tuple( 2.0, 4.0, 6.0) );
			point.SetWeight( 3.0 )  ;
			point.SetWeight2( 9.0 ) ;
			point.SetError(3.0) ;
			data = point.GetData();

			REQUIRE( thrust::get<0>(data) ==  Approx(3.0) );
			REQUIRE( thrust::get<1>(data) ==  Approx(9.0) );
			REQUIRE( thrust::get<2>(data) ==  Approx(3.0) );
			REQUIRE( thrust::get<3>(data) ==  Approx(2.0) );
			REQUIRE( thrust::get<4>(data) ==  Approx(4.0) );
			REQUIRE( thrust::get<5>(data) ==  Approx(6.0) );

		}

	}

}
