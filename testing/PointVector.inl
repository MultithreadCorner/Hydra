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
#include <algorithm>

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Point.h>
#include <hydra/PointVector.h>
#include <thrust/tuple.h>


#include <array>
TEST_CASE( "PointVector","PointVector 1D case" ) {

	typedef  Point<double, 3> Point_t;
	typedef  PointVector<Point_t, hydra::host> PointVector_h;
	typedef  PointVector<Point_t, hydra::device> PointVector_d;

	SECTION( "Constructors " )
	{
		SECTION( "default constructor" )
								{
			auto points = PointVector_h();

			REQUIRE( points.size() ==  0.0 );


								}

		SECTION( "n-constructor" )
		{

			auto points = PointVector_h(10);

			REQUIRE( points.size() ==  10 );


		}

		SECTION( "range semantics, std::fill and copy constructor same backends" )
		{

			typedef typename PointVector_h::point_t point_t;

			auto points = PointVector_h(10);
			std::fill(points.begin(), points.end(), point_t({1.0, 2.0, 3.0},1.5).GetData());

			for(auto row: points)
				REQUIRE( (point_t) row ==  point_t({1.0, 2.0, 3.0},1.5 ));

			PointVector_h points2(points);

			for(size_t i=0; i<points.size() ; i++)
				REQUIRE( points[i] ==  points2[i]);

		}

		SECTION( "range semantics, std::fill and copy constructor different backends" )
		{

			typedef typename PointVector_h::point_t point_t;

			auto points = PointVector_h(10);
			std::fill(points.begin(), points.end(), point_t({1.0, 2.0, 3.0},1.5).GetData());

			for(auto row: points)
				REQUIRE( (point_t) row ==  point_t({1.0, 2.0, 3.0},1.5 ));

			PointVector_d points2(points);

			for(size_t i=0; i<points.size() ; i++)
				REQUIRE( points[i] ==  points2[i]);

		}

	}

	SECTION( "Adding, getting and modifying points" )
	{

		SECTION( "adding and getting points" )
		{
			typedef typename PointVector_h::point_t point_t;

			PointVector_h  points;
			for(size_t i=0; i<10 ; i++)
			{
				points.AddPoint( point_t({i+ 1.0, i+ 2.0, i+ 3.0}, i ) );
			}

			/**
			 * getting points via subscript iterator with type conversion
			 */
			for(size_t i=0; i<points.size() ; i++)
			{
				REQUIRE( (point_t) points[i] == point_t({i+ 1.0, i+ 2.0, i+ 3.0}, i ) );
			}

			/**
			 * getting points via GetPoint
			 */
			for(size_t i=0; i<points.size() ; i++)
			{
				REQUIRE( points.GetPoint(i) == point_t({i+ 1.0, i+ 2.0, i+ 3.0}, i ) );
			}

		}


	}

	SECTION( "Adding, getting and modifying points with subscript operator" )
		{

			SECTION( "adding and getting points" )
			{
				typedef typename PointVector_h::point_t point_t;

				PointVector_h  points(10);
				for(size_t i=0; i<10 ; i++)
				{
					points[i] = point_t({i+ 1.0, i+ 2.0, i+ 3.0}, i ).GetData() ;
				}

				/**
				 * getting points via subscript iterator with type conversion
				 */
				for(size_t i=0; i<points.size() ; i++)
				{
					REQUIRE( (point_t) points[i] == point_t({i+ 1.0, i+ 2.0, i+ 3.0}, i ) );
				}

				/**
				 * getting points via GetPoint
				 */
				for(size_t i=0; i<points.size() ; i++)
				{
					REQUIRE( points.GetPoint(i) == point_t({i+ 1.0, i+ 2.0, i+ 3.0}, i ) );
				}

			}


		}

	SECTION( "lambdas..." )
		{

			SECTION( "std::for_each on host" )
			{
				typedef typename PointVector_h::point_t point_t;

				PointVector_h  points(10);
				std::for_each(points.begin(), points.end(),
						[]__host__ __device__ (PointVector_h::reference point){ point=point_t({1.0, 2.0, 3.0}, 1.5 ).GetData(); });

				/**
				 * via GetPoint
				 */
				for(size_t i=0; i<points.size() ; i++)
				{
					REQUIRE( points.GetPoint(i) == point_t({1.0, 2.0, 3.0}, 1.5 ) );
				}

			}

			SECTION( "thrust::for_each on device" )
			{
				typedef typename PointVector_d::point_t point_t;

				PointVector_d  points(10);
				thrust::for_each(points.begin(), points.end(),
						[]__host__ __device__(PointVector_h::reference point)
				{ point=point_t({1.0, 2.0, 3.0}, 1.5 ).GetData(); });

				/**
				 * via GetPoint
				 */
				for(size_t i=0; i<points.size() ; i++)
				{
					REQUIRE( points.GetPoint(i) == point_t({1.0, 2.0, 3.0}, 1.5 ) );
				}

			}
		}

}
