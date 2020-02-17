/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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
 * lambda.inl
 *
 *  Created on: 17/02/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef LAMBDA_TEST_INL_
#define LAMBDA_TEST_INL_

#include <catch/catch.hpp>


#include <hydra/multivector.h>
#include <hydra/Tuple.h>
#include <hydra/Lambda.h>
#include <hydra/Parameter.h>
#include <hydra/device/System.h>
#include <hydra/host/System.h>

//named-like function arguments
using namespace hydra::arguments;

declarg(U_arg, double)
declarg(V_arg, double)
declarg(X_arg, double)
declarg(Y_arg, double)
declarg(Z_arg, double)

TEST_CASE( "lambda host","hydra::Lambda<Type,0>" )
{

	typedef hydra::tuple<X_arg, Y_arg, Z_arg>               right_tuple;
	typedef hydra::tuple<U_arg, V_arg,  X_arg, Y_arg, Z_arg> wrong_tuple;

	typedef hydra::multivector<	right_tuple, hydra::host::sys_t> right_table_d;
	typedef hydra::multivector<	wrong_tuple, hydra::host::sys_t> wrong_table_d;


		SECTION( "Call lambda: native signature" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple( a*a, b*b, c*c );
			});

			X_arg x = 2.0;
			Y_arg y = 4.0;
			Z_arg z = 8.0;

			auto result = lambda(x, y, z);

			REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
			REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );

		}

		SECTION( "Call lambda: native signature getting by type" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple( a*a, b*b, c*c );
			});

			X_arg x = 2.0;
			Y_arg y = 4.0;
			Z_arg z = 8.0;

			auto result = lambda(x, y, z);

			REQUIRE( hydra::get<X_arg>(result) == Approx( 4.0  ) );
			REQUIRE( hydra::get<Y_arg>(result) == Approx( 16.0 ) );
			REQUIRE( hydra::get<Z_arg>(result) == Approx( 64.0 ) );

		}

		SECTION( "Call lambda: tuple with native signature" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple( a*a, b*b, c*c);
			});

			auto result = lambda( right_tuple(2.0, 4.0, 8.0) );

			REQUIRE( hydra::get<X_arg>(result) == Approx( 4.0  ) );
			REQUIRE( hydra::get<Y_arg>(result) == Approx( 16.0 ) );
			REQUIRE( hydra::get<Z_arg>(result) == Approx( 64.0 ) );

		}

		SECTION( "Call lambda: container entries matching signature" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});

			right_table_d table(10,  hydra::make_tuple(2.0, 4.0, 8.0));

			for( auto entry: table)
			{
				auto result = lambda( entry );

				REQUIRE( hydra::get<X_arg>(result) == Approx( 4.0  ) );
				REQUIRE( hydra::get<Y_arg>(result) == Approx( 16.0 ) );
				REQUIRE( hydra::get<Z_arg>(result) == Approx( 64.0 ) );

			}

		}


		SECTION( "Call lambda: tuple not matching native signature" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple( a*a, b*b, c*c );
			});

			auto result = lambda( wrong_tuple( 1.0, 1.0, 2.0, 4.0, 8.0) );

			REQUIRE( hydra::get<X_arg>(result) == Approx( 4.0  ) );
			REQUIRE( hydra::get<Y_arg>(result) == Approx( 16.0 ) );
			REQUIRE( hydra::get<Z_arg>(result) == Approx( 64.0 ) );

		}

		SECTION( "Call lambda: container entries not matching signature" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});

			wrong_table_d table(10,  hydra::make_tuple(1.0, 1.0, 2.0, 4.0, 8.0));

			for( auto entry: table)
			{
				auto result = lambda( entry );

				REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
				REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );

			}

		}

}

TEST_CASE( "lambda device","hydra::Lambda<Type,0>" )
{

	typedef hydra::tuple<X_arg, Y_arg, Z_arg>               right_tuple;
	typedef hydra::tuple<U_arg, V_arg,  X_arg, Y_arg, Z_arg> wrong_tuple;

	typedef hydra::multivector<	right_tuple, hydra::device::sys_t> right_table_d;
	typedef hydra::multivector<	wrong_tuple, hydra::device::sys_t> wrong_table_d;


		SECTION( "Call lambda: native signature" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple( a*a, b*b, c*c );
			});

			X_arg x = 2.0;
			Y_arg y = 4.0;
			Z_arg z = 8.0;

			auto result = lambda(x, y, z);

			REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
			REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );

		}

		SECTION( "Call lambda: native signature getting by type" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple( a*a, b*b, c*c );
			});

			X_arg x = 2.0;
			Y_arg y = 4.0;
			Z_arg z = 8.0;

			auto result = lambda(x, y, z);

			REQUIRE( hydra::get<X_arg>(result) == Approx( 4.0  ) );
			REQUIRE( hydra::get<Y_arg>(result) == Approx( 16.0 ) );
			REQUIRE( hydra::get<Z_arg>(result) == Approx( 64.0 ) );

		}

		SECTION( "Call lambda: tuple with native signature" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple( a*a, b*b, c*c);
			});

			auto result = lambda( right_tuple(2.0, 4.0, 8.0) );

			REQUIRE( hydra::get<X_arg>(result) == Approx( 4.0  ) );
			REQUIRE( hydra::get<Y_arg>(result) == Approx( 16.0 ) );
			REQUIRE( hydra::get<Z_arg>(result) == Approx( 64.0 ) );

		}

		SECTION( "Call lambda: container entries matching signature" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});

			right_table_d table(10,  hydra::make_tuple(2.0, 4.0, 8.0));

			for( auto entry: table)
			{
				auto result = lambda( entry );

				REQUIRE( hydra::get<X_arg>(result) == Approx( 4.0  ) );
				REQUIRE( hydra::get<Y_arg>(result) == Approx( 16.0 ) );
				REQUIRE( hydra::get<Z_arg>(result) == Approx( 64.0 ) );

			}

		}


		SECTION( "Call lambda: tuple not matching native signature" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple( a*a, b*b, c*c );
			});

			auto result = lambda( wrong_tuple( 1.0, 1.0, 2.0, 4.0, 8.0) );

			REQUIRE( hydra::get<X_arg>(result) == Approx( 4.0  ) );
			REQUIRE( hydra::get<Y_arg>(result) == Approx( 16.0 ) );
			REQUIRE( hydra::get<Z_arg>(result) == Approx( 64.0 ) );

		}

		SECTION( "Call lambda: container entries not matching signature" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});

			wrong_table_d table(10,  hydra::make_tuple(1.0, 1.0, 2.0, 4.0, 8.0));

			for( auto entry: table)
			{
				auto result = lambda( entry );

				REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
				REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );

			}

		}

}
#endif /* LAMBDA_TEST_INL_ */
