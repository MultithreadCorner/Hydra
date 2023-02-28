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

/*
 * Test wrapping and evaluation of c++ lambdas using
 * named arguments in host backend (cpp) using host-compiler
 * and nvcc.
 *
 */
TEST_CASE( "lambda host","hydra::Lambda<Type,0>" )
{

	typedef hydra::tuple<X_arg, Y_arg, Z_arg>                     right_tuple;
	typedef hydra::tuple<U_arg, V_arg, X_arg, Y_arg, Z_arg>       wrong_tuple;
	typedef hydra::tuple<U_arg, V_arg, X_arg>                 x_partial_tuple;
	typedef hydra::tuple<Y_arg, Z_arg>                       yz_partial_tuple;

	typedef hydra::host::vector<V_arg>                       v_vector;
	typedef hydra::host::vector<X_arg>                       x_vector;
	typedef hydra::host::vector<Y_arg>                       y_vector;
	typedef hydra::host::vector<Z_arg>                       z_vector;

	typedef hydra::multivector<	right_tuple, hydra::host::sys_t> right_table;
	typedef hydra::multivector<	wrong_tuple, hydra::host::sys_t> wrong_table;
	typedef hydra::multivector< x_partial_tuple,  hydra::host::sys_t>  x_partial_table;
	typedef hydra::multivector< yz_partial_tuple, hydra::host::sys_t> yz_partial_table;

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

			right_table table(10,  hydra::make_tuple(2.0, 4.0, 8.0));

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

			wrong_table table(10,  hydra::make_tuple(1.0, 1.0, 2.0, 4.0, 8.0));

			for( auto entry: table)
			{
				auto result = lambda( entry );

				REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
				REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );

			}

		}

		SECTION( "Call lambda: two complementary containers with lambda arguments" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});

			x_partial_table   x_partial(10, hydra::make_tuple(1.0, 1.0, 2.0));
			yz_partial_table yz_partial(10, hydra::make_tuple(4.0, 8.0));


			for(size_t i=0; i<10;i++)
			{
				auto result = lambda( x_partial[i],  yz_partial[i]);

				REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
				REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );
			}
		}


		SECTION( "Call lambda: two complementary tuples with lambda arguments" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});


			auto result = lambda( x_partial_tuple(1.0, 1.0, 2.0),  yz_partial_tuple(4.0, 8.0));

			REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
			REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );

		}



		SECTION( "Call lambda: one variable and one tuple containing lambda arguments" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});


			auto result = lambda( X_arg(2.0),  yz_partial_tuple(4.0, 8.0));

			REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
			REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );

		}

		SECTION( "Call lambda: one variable and one tuple containing lambda arguments(inverse order)" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});


			auto result = lambda(yz_partial_tuple(4.0, 8.0) , X_arg(2.0));

			REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
			REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );

		}


		SECTION( "Call lambda:  two complementary containers with lambda arguments" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});

			x_vector           x_vector(10, 2.0);
			yz_partial_table yz_partial(10, hydra::make_tuple(4.0, 8.0));


			for(size_t i=0; i<10;i++)
			{
				auto result = lambda( x_vector[i],  yz_partial[i]);

				REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
				REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );
			}
		}

		SECTION( "Call lambda:  two complementary containers with lambda arguments" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});

			x_vector           x_vector(10, 2.0);
			yz_partial_table yz_partial(10, hydra::make_tuple(4.0, 8.0));


			for(size_t i=0; i<10;i++)
			{
				auto result = lambda( yz_partial[i], x_vector[i]);

				REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
				REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );
			}
		}

/*
 * FIXME: activate this test when finish implementation
 *
		SECTION( "Call lambda:  two complementary containers with lambda arguments" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});

			x_vector           x_vector(10, 2.0);
			y_vector           y_vector(10, 4.0);
			z_vector           z_vector(10, 8.0);
			v_vector           v_vector(10, 1.0);

            auto table = hydra::zip(x_vector, v_vector, z_vector, y_vector);

			for(auto entry: table)
			{
				auto result = lambda(entry );

				REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
				REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );
			}
		}
*/
}

/*
 * Test wrapping and evaluation of c++ lambdas using
 * named arguments in device backend (cpp, omp, ) using host-compiler
 * and nvcc.
 *
 */
TEST_CASE( "lambda device","hydra::Lambda<Type,0>" )
{

	typedef hydra::tuple<X_arg, Y_arg, Z_arg>                     right_tuple;
	typedef hydra::tuple<U_arg, V_arg,  X_arg, Y_arg, Z_arg>      wrong_tuple;
	typedef hydra::tuple<U_arg, V_arg, X_arg>                 x_partial_tuple;
	typedef hydra::tuple<Y_arg, Z_arg>                       yz_partial_tuple;

	typedef hydra::device::vector<V_arg>                       v_vector;
	typedef hydra::device::vector<X_arg>                       x_vector;
	typedef hydra::device::vector<Y_arg>                       y_vector;
	typedef hydra::device::vector<Z_arg>                       z_vector;

	typedef hydra::multivector<	right_tuple, hydra::device::sys_t> right_table;
	typedef hydra::multivector<	wrong_tuple, hydra::device::sys_t> wrong_table;
	typedef hydra::multivector< x_partial_tuple,  hydra::device::sys_t>  x_partial_table;
	typedef hydra::multivector< yz_partial_tuple, hydra::device::sys_t> yz_partial_table;

		SECTION( "Call lambda: native signature" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple( a*a, b*b, c*c );
			});

			X_arg x = 2.0;
			Y_arg y = 4.0;
			Z_arg z = 8.0;

			hydra::multivector<hydra::tuple<double,double>,hydra::device::sys_t> dataset(10, hydra::make_tuple(1.0, 2.0));


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

			right_table table(10,  hydra::make_tuple(2.0, 4.0, 8.0));

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

			wrong_table table(10,  hydra::make_tuple(1.0, 1.0, 2.0, 4.0, 8.0));

			for( auto entry: table)
			{
				auto result = lambda( entry );

				REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
				REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );

			}

		}


		SECTION( "Call lambda: two complementary containers with lambda arguments" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});

			x_partial_table   x_partial(10, hydra::make_tuple(1.0, 1.0, 2.0));
			yz_partial_table yz_partial(10, hydra::make_tuple(4.0, 8.0));


			for(size_t i=0; i<10;i++)
			{
				auto result = lambda( x_partial[i],  yz_partial[i]);

				REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
				REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );
			}
		}


		SECTION( "Call lambda: two complementary tuples with lambda arguments" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});


			auto result = lambda( x_partial_tuple(1.0, 1.0, 2.0),  yz_partial_tuple(4.0, 8.0));

			REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
			REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );

		}



		SECTION( "Call lambda: one variable and one tuple containing lambda arguments" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});


			auto result = lambda( X_arg(2.0),  yz_partial_tuple(4.0, 8.0));

			REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
			REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );

		}

		SECTION( "Call lambda: one variable and one tuple containing lambda arguments(inverse order)" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});


			auto result = lambda(yz_partial_tuple(4.0, 8.0) , X_arg(2.0));

			REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
			REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );

		}


		SECTION( "Call lambda:  two complementary containers with lambda arguments" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});

			x_vector           x_vector(10, 2.0);
			yz_partial_table yz_partial(10, hydra::make_tuple(4.0, 8.0));


			for(size_t i=0; i<10;i++)
			{
				auto result = lambda( x_vector[i],  yz_partial[i]);

				REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
				REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );
			}
		}

		SECTION( "Call lambda:  two complementary containers with lambda arguments" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});

			x_vector           x_vector(10, 2.0);
			yz_partial_table yz_partial(10, hydra::make_tuple(4.0, 8.0));


			for(size_t i=0; i<10;i++)
			{
				auto result = lambda( yz_partial[i], x_vector[i]);

				REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
				REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );
			}
		}

/*
 * FIXME: activate this test when finish implementation
 *
		SECTION( "Call lambda:  two complementary containers with lambda arguments" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});

			x_vector           x_vector(10, 2.0);
			y_vector           y_vector(10, 4.0);
			z_vector           z_vector(10, 8.0);
			v_vector           v_vector(10, 1.0);

            auto table = hydra::zip(x_vector, v_vector, z_vector, y_vector);

			for(auto entry: table)
			{
				auto result = lambda(entry );

				REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
				REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );
			}
		}
*/
}


/*
 * Test wrapping and evaluation of c++ parameterized  lambdas using
 * named arguments in host backend (cpp) using host-compiler
 * and nvcc.
 *
 */
TEST_CASE( "parameterized lambda host","hydra::Lambda<Type,2>" )
{

	typedef hydra::tuple<X_arg, Y_arg, Z_arg>                     right_tuple;
	typedef hydra::tuple<U_arg, V_arg,  X_arg, Y_arg, Z_arg>      wrong_tuple;
	typedef hydra::tuple<U_arg, V_arg, X_arg>                 x_partial_tuple;
	typedef hydra::tuple<Y_arg, Z_arg>                       yz_partial_tuple;

	typedef hydra::host::vector<V_arg>                       v_vector;
	typedef hydra::host::vector<X_arg>                       x_vector;
	typedef hydra::host::vector<Y_arg>                       y_vector;
	typedef hydra::host::vector<Z_arg>                       z_vector;

	typedef hydra::multivector<	right_tuple, hydra::host::sys_t> right_table;
	typedef hydra::multivector<	wrong_tuple, hydra::host::sys_t> wrong_table;
	typedef hydra::multivector< x_partial_tuple,  hydra::host::sys_t>  x_partial_table;
	typedef hydra::multivector< yz_partial_tuple, hydra::host::sys_t> yz_partial_table;

		SECTION( "Call lambda: native signature" )
		{
			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			X_arg x = 2.0;
			Y_arg y = 4.0;
			Z_arg z = 8.0;

			auto result = lambda(x, y, z);

			REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
			REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

		}

		SECTION( "Call lambda: native signature getting by type" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			X_arg x = 2.0;
			Y_arg y = 4.0;
			Z_arg z = 8.0;

			auto result = lambda(x, y, z);

			REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
			REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

		}

		SECTION( "Call lambda: tuple with native signature" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			auto result = lambda( right_tuple(2.0, 4.0, 8.0) );

			REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
			REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

		}

		SECTION( "Call lambda: container entries matching signature" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			right_table table(10,  hydra::make_tuple(2.0, 4.0, 8.0));

			for( auto entry: table)
			{
				auto result = lambda( entry );

				REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
				REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

			}

		}


		SECTION( "Call lambda: tuple not matching native signature" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			auto result = lambda( wrong_tuple( 1.0, 1.0, 2.0, 4.0, 8.0) );

			REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
			REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

		}

		SECTION( "Call lambda: container entries not matching signature" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			wrong_table table(10,  hydra::make_tuple(1.0, 1.0, 2.0, 4.0, 8.0));

			for( auto entry: table)
			{
				auto result = lambda( entry );

				REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
				REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

			}

		}

		SECTION( "Call lambda: two complementary containers with lambda arguments" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			x_partial_table   x_partial(10, hydra::make_tuple(1.0, 1.0, 2.0));
			yz_partial_table yz_partial(10, hydra::make_tuple(4.0, 8.0));


			for(size_t i=0; i<10;i++)
			{
				auto result = lambda( x_partial[i],  yz_partial[i]);

				REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
				REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

			}
		}


		SECTION( "Call lambda: two complementary tuples with lambda arguments" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;


			auto result = lambda( x_partial_tuple(1.0, 1.0, 2.0),  yz_partial_tuple(4.0, 8.0));

			REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
			REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

		}



		SECTION( "Call lambda: one variable and one tuple containing lambda arguments" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;


			auto result = lambda( X_arg(2.0),  yz_partial_tuple(4.0, 8.0));

			REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
			REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

		}

		SECTION( "Call lambda: one variable and one tuple containing lambda arguments(inverse order)" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;


			auto result = lambda(yz_partial_tuple(4.0, 8.0) , X_arg(2.0));

			REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
			REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

		}


		SECTION( "Call lambda:  two complementary containers with lambda arguments" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			x_vector           x_vector(10, 2.0);
			yz_partial_table yz_partial(10, hydra::make_tuple(4.0, 8.0));


			for(size_t i=0; i<10;i++)
			{
				auto result = lambda( x_vector[i],  yz_partial[i]);

				REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
				REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

			}
		}

		SECTION( "Call lambda:  two complementary containers with lambda arguments" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			x_vector           x_vector(10, 2.0);
			yz_partial_table yz_partial(10, hydra::make_tuple(4.0, 8.0));


			for(size_t i=0; i<10;i++)
			{
				auto result = lambda( yz_partial[i], x_vector[i]);

				REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
				REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

			}
		}

}

/*
 * Test wrapping and evaluation of c++ lambdas using
 * named arguments in device backend (cpp, omp, ) using host-compiler
 * and nvcc.
 *
 */
TEST_CASE(  "parameterized lambda device","hydra::Lambda<Type,2>" )
{

	typedef hydra::tuple<X_arg, Y_arg, Z_arg>                     right_tuple;
	typedef hydra::tuple<U_arg, V_arg,  X_arg, Y_arg, Z_arg>      wrong_tuple;
	typedef hydra::tuple<U_arg, V_arg, X_arg>                 x_partial_tuple;
	typedef hydra::tuple<Y_arg, Z_arg>                       yz_partial_tuple;

	typedef hydra::device::vector<V_arg>                       v_vector;
	typedef hydra::device::vector<X_arg>                       x_vector;
	typedef hydra::device::vector<Y_arg>                       y_vector;
	typedef hydra::device::vector<Z_arg>                       z_vector;

	typedef hydra::multivector<	right_tuple, hydra::device::sys_t> right_table;
	typedef hydra::multivector<	wrong_tuple, hydra::device::sys_t> wrong_table;
	typedef hydra::multivector< x_partial_tuple,  hydra::device::sys_t>  x_partial_table;
	typedef hydra::multivector< yz_partial_tuple, hydra::device::sys_t> yz_partial_table;

		SECTION( "Call lambda: native signature" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			X_arg x = 2.0;
			Y_arg y = 4.0;
			Z_arg z = 8.0;

			auto result = lambda(x, y, z);

			REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
			REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

		}

		SECTION( "Call lambda: native signature getting by type" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			X_arg x = 2.0;
			Y_arg y = 4.0;
			Z_arg z = 8.0;

			auto result = lambda(x, y, z);

			REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
			REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

		}

		SECTION( "Call lambda: tuple with native signature" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			auto result = lambda( right_tuple(2.0, 4.0, 8.0) );

			REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
			REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

		}

		SECTION( "Call lambda: container entries matching signature" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			right_table table(10,  hydra::make_tuple(2.0, 4.0, 8.0));

			for( auto entry: table)
			{
				auto result = lambda( entry );

				REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
				REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

			}

		}


		SECTION( "Call lambda: tuple not matching native signature" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			auto result = lambda( wrong_tuple( 1.0, 1.0, 2.0, 4.0, 8.0) );

			REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
			REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

		}

		SECTION( "Call lambda: container entries not matching signature" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			wrong_table table(10,  hydra::make_tuple(1.0, 1.0, 2.0, 4.0, 8.0));

			for( auto entry: table)
			{
				auto result = lambda( entry );

				REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
				REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

			}

		}


		SECTION( "Call lambda: two complementary containers with lambda arguments" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			x_partial_table   x_partial(10, hydra::make_tuple(1.0, 1.0, 2.0));
			yz_partial_table yz_partial(10, hydra::make_tuple(4.0, 8.0));


			for(size_t i=0; i<10;i++)
			{
				auto result = lambda( x_partial[i],  yz_partial[i]);

				REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
				REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

			}
		}


		SECTION( "Call lambda: two complementary tuples with lambda arguments" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;


			auto result = lambda( x_partial_tuple(1.0, 1.0, 2.0),  yz_partial_tuple(4.0, 8.0));

			REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
			REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

		}



		SECTION( "Call lambda: one variable and one tuple containing lambda arguments" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;


			auto result = lambda( X_arg(2.0),  yz_partial_tuple(4.0, 8.0));

			REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
			REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

		}

		SECTION( "Call lambda: one variable and one tuple containing lambda arguments(inverse order)" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;


			auto result = lambda(yz_partial_tuple(4.0, 8.0) , X_arg(2.0));

			REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
			REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
			REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

		}


		SECTION( "Call lambda:  two complementary containers with lambda arguments" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			x_vector           x_vector(10, 2.0);
			yz_partial_table yz_partial(10, hydra::make_tuple(4.0, 8.0));


			for(size_t i=0; i<10;i++)
			{
				auto result = lambda( x_vector[i],  yz_partial[i]);

				REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
				REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );
			}
		}

		SECTION( "Call lambda:  two complementary containers with lambda arguments" )
		{

			auto c0 = hydra::Parameter::Create("c0").Value(0.0);
			auto c1 = hydra::Parameter::Create("c1").Value(0.0);

			auto lambda = hydra::wrap_lambda(
					[]__hydra_dual__(size_t n, const hydra::Parameter* pars, X_arg a, Y_arg b, Z_arg c){

				double c0 = pars[0];
				double c1 = pars[1];

				return right_tuple( c0*a + c1, c0*b + c1, c0*c + c1 );
			}, c0, c1 );

			lambda.Parameter("c0")= 2.0;
			lambda.Parameter("c1")= 1.0;

			x_vector           x_vector(10, 2.0);
			yz_partial_table yz_partial(10, hydra::make_tuple(4.0, 8.0));


			for(size_t i=0; i<10;i++)
			{
				//decltype(x_vector[i])::dummy a;

				auto result = lambda( yz_partial[i], x_vector[i]);

				REQUIRE( hydra::get<0>(result) == Approx(  5.0 ) );
				REQUIRE( hydra::get<1>(result) == Approx(  9.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 17.0 ) );

			}
		}

/*
 * FIXME: activate this test when finish implementation
 *
		SECTION( "Call lambda:  two complementary containers with lambda arguments" )
		{
			auto lambda = hydra::wrap_lambda( []__hydra_dual__(X_arg a, Y_arg b, Z_arg c){

				return right_tuple(a*a,  b*b, c*c);
			});

			x_vector           x_vector(10, 2.0);
			y_vector           y_vector(10, 4.0);
			z_vector           z_vector(10, 8.0);
			v_vector           v_vector(10, 1.0);

            auto table = hydra::zip(x_vector, v_vector, z_vector, y_vector);

			for(auto entry: table)
			{
				auto result = lambda(entry );

				REQUIRE( hydra::get<0>(result) == Approx( 4.0  ) );
				REQUIRE( hydra::get<1>(result) == Approx( 16.0 ) );
				REQUIRE( hydra::get<2>(result) == Approx( 64.0 ) );
			}
		}
*/
}


#endif /* LAMBDA_TEST_INL_ */
